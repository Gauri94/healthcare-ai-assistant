"""The web API. Start it with:   uv run fastapi dev app/main.py
Then open http://127.0.0.1:8000/docs for interactive docs (Swagger UI).

CONCEPT (FastAPI): FastAPI is an ASYNC web framework. It runs on uvicorn, an
ASGI server that owns the EVENT LOOP (that's why we never call asyncio.run here).
Each incoming request becomes a task on that loop, so while one member's request
awaits a slow tool, the server keeps handling other members' requests.
This is why all the M1 rules matter: one blocking call here stalls EVERYONE.
"""

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Path, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app import graph_agent, llm_agent
from app.auth import CurrentMember
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, ClaimOut, CoverageOut, DeductibleOut
from app.tools import ToolError, check_coverage, get_claim_status, get_deductible, run_tool

# Show INFO-level logs (e.g. "LLM fallback: ...") in the terminal.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# CONCEPT (FastAPI): a LIFESPAN function runs setup code before the server starts
# taking requests (before `yield`) and cleanup code when it shuts down (after
# `yield`). Typical uses: open/close DB pools and HTTP clients, load ML models
# into memory once (we'll load the router and emergency models here in M6/M7).
# PYTHON: @asynccontextmanager turns an async generator (a function with `yield`)
# into an `async with`-style context manager, which is what FastAPI expects.
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    yield                              # the server runs while we're paused here
    await llm_agent.close_client()     # shutdown: close LLM connections cleanly


app = FastAPI(
    lifespan=lifespan,
    title="Healthcare AI Assistant",
    version="0.3.0",
    description="Member-facing health plan assistant (synthetic data only). "
                "Click **Authorize** and paste `token-asha` or `token-ben` to log in.",
)


# CONCEPT (FastAPI): an EXCEPTION HANDLER turns a Python exception into an HTTP
# response in ONE place, instead of try/except in every endpoint. Any endpoint
# whose tool raises ToolError automatically returns e.g.
#   404 {"detail": "No claim C-2001 found on your account."}
# or 504 for a timeout (status_code comes from the ToolError itself).
@app.exception_handler(ToolError)
async def tool_error_handler(request: Request, exc: ToolError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


# Opening the bare server URL (http://127.0.0.1:8000/) used to show
# {"detail": "Not Found"} because no route matched "/". Send visitors to the docs.
# include_in_schema=False keeps this helper route out of the /docs page itself.
# CONCEPT (HTTP): a 307 redirect tells the browser "go to this other URL instead".
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


# PYTHON: @app.get("/health") is a DECORATOR. It registers the function below as
# the handler for GET requests to /health. The function's return value (a dict)
# is converted to JSON automatically.
#
# CONCEPT (deployment): load balancers and Kubernetes ping /health to decide
# whether this server is alive. Deliberately no auth and no slow work here.
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ---- Direct endpoints: the same tools, without chat ----
# Handy for testing tools in isolation, and they show FastAPI basics:
# path parameters, validation and response models.
#
# SECURITY: the URLs say "/me/...", never "/members/{member_id}/...". With an ID in
# the URL, changing M123 to M456 might show someone else's data. That bug is
# called IDOR (Insecure Direct Object Reference) and it's one of the most common
# API vulnerabilities. Here the member is always taken from the token.

# CONCEPT: response_model=DeductibleOut makes FastAPI validate AND filter the
# output to exactly that model's fields, and documents it in /docs.
@app.get("/me/deductible", response_model=DeductibleOut)
async def my_deductible(member_id: CurrentMember):
    # `member_id: CurrentMember` runs the auth dependency first (app/auth.py).
    # No valid token -> 401 and this function never runs.
    return await run_tool(get_deductible(member_id))


@app.get("/me/claims/{claim_id}", response_model=ClaimOut)
async def my_claim(
    # CONCEPT: {claim_id} in the URL becomes this PATH PARAMETER. Path(pattern=...)
    # validates it with a regex: "/me/claims/abc" gets a 422 before any tool runs.
    claim_id: Annotated[str, Path(pattern=r"^C-\d{4}$", examples=["C-1001"])],
    member_id: CurrentMember,
):
    return await run_tool(get_claim_status(member_id, claim_id))


@app.get("/coverage/{cpt_code}", response_model=CoverageOut)
async def coverage(
    cpt_code: Annotated[str, Path(pattern=r"^\d{5}$", examples=["73721"])],
    member_id: CurrentMember,
):
    return await run_tool(check_coverage(member_id, cpt_code))


# ---- The chat endpoint: the assistant itself ----
# CONCEPT: `body: ChatRequest` tells FastAPI to read the JSON request body and
# validate it with Pydantic. Bad input (empty message, too long, unknown fields
# like member_id) -> automatic 422 with details, and this function never runs.
@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, member_id: CurrentMember) -> ChatResponse:
    # Pick the engine from settings. Both fall back to the rule-based planner by
    # themselves when needed (no key, budget reached, provider error).
    if get_settings().agent_engine == "langgraph":
        result = await graph_agent.answer(member_id, body.message, body.conversation_id)
    else:
        result = await llm_agent.answer(member_id, body.message)
    # NOTE: returning tool_calls (including raw results) is great for debugging and
    # evals. In production you might hide results from the client and keep them
    # only in server-side traces.
    return ChatResponse(
        reply=result.reply,
        tool_calls=result.tool_calls,
        mode=result.mode,
        fallback_reason=result.fallback_reason,
        cost_usd=round(result.cost_usd, 6),
        conversation_id=result.conversation_id,
        request_id=str(uuid.uuid4()),
    )

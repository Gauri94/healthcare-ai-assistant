"""Request and response shapes for the API, defined with Pydantic.

CONCEPT (Pydantic): a BaseModel subclass describes the shape of some data with
type hints. Pydantic then:
  - VALIDATES incoming data (wrong type or too long -> automatic 422 error with
    a clear message, before your code ever runs),
  - CONVERTS types where it safely can (e.g. "240" -> 240.0 for a float field),
  - SERIALIZES outgoing data to JSON,
  - and generates a JSON SCHEMA, which FastAPI turns into the /docs page.

CONCEPT (agents): the same idea powers LLM tool calling and structured output:
you describe a schema, and the model's output is validated against it (M3).

SECURITY: notice that NO request model has a member_id field. Identity comes
only from the login token (app/auth.py).
"""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    # CONCEPT: extra="forbid" rejects unknown fields with a 422. If a client tries
    # to sneak in {"message": "...", "member_id": "M456"}, the request is refused
    # instead of the field being silently ignored. Loud failure = easier to spot abuse.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"message": "Is a knee MRI (73721) covered, and how much deductible do I have left?"}]
        },
    )

    # PYTHON/PYDANTIC: Field(...) adds validation rules and docs to one field.
    # max_length caps input size: protects cost (LLM tokens) and blocks giant
    # prompt-injection payloads.
    message: str = Field(min_length=1, max_length=2000, description="What the member typed.")


class ToolCall(BaseModel):
    """A record of one tool the assistant ran while answering. Useful for debugging,
    tests and evals ("did it call the right tool?")."""

    name: str
    args: dict[str, str]
    ok: bool
    # PYTHON: `dict | None = None` means "a dict, or None, defaulting to None",
    # i.e. an OPTIONAL field. A successful call fills result; a failed one fills error.
    result: dict | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCall]
    # "llm" = answered by the LLM agent; "fallback" = rule-based planner (no key,
    # budget reached, or provider error). fallback_reason says which.
    mode: str
    fallback_reason: str | None = None
    # CONCEPT (cost observability): what this answer cost. Logged per request, it
    # tells you cost per conversation, which drives model choice and pricing.
    cost_usd: float
    # CONCEPT (observability): a unique ID per request. It goes in logs and traces
    # too, so when a member reports a bad answer, support can find the exact request.
    request_id: str


# ---- Response models for the direct (non-chat) endpoints ----
# CONCEPT: response_model in FastAPI FILTERS the output to exactly these fields.
# Even if a tool accidentally returned extra internal data, it wouldn't leak to
# the client. Another layer of data minimization.

class DeductibleOut(BaseModel):
    deductible_total: float
    deductible_met: float
    remaining: float


class ClaimOut(BaseModel):
    claim_id: str
    service: str
    status: str
    billed: float
    member_owes: float | None  # None while the claim is still pending


class CoverageOut(BaseModel):
    cpt_code: str
    covered: bool
    prior_auth: bool | None = None
    coinsurance: float | None = None
    copay: float | None = None
    note: str | None = None

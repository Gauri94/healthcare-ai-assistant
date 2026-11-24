"""The tools the agent can call.

CONCEPT (agents): a TOOL is a normal function the LLM can ask us to run. The LLM
only *chooses* a tool and its arguments; our code runs it and returns the result.
So the LLM never touches the data directly, and these functions are where
facts, identity and security are enforced.

CONCEPT (async): tools are `async` because real ones wait on slow I/O (a claims
API, a database, a vector search). While one tool waits, the event loop can run
other tools or serve other members' requests on the same server.
"""

import asyncio  # Python's built-in library for async code: event loop, sleep, gather, timeout
from collections.abc import Coroutine  # type of an un-awaited coroutine (used by run_tool)
from typing import Any

from app import data  # works because app/__init__.py makes "app" a package


# CONCEPT: a CUSTOM EXCEPTION. Subclassing Exception gives us our own error type,
# so the agent can tell two kinds of failure apart:
#   - ToolError  -> expected ("claim not found"): explain it politely to the member
#   - anything else (KeyError, TypeError...) -> a bug: log it, show a generic error
class ToolError(Exception):
    """A tool failed in a way the agent should explain to the member."""

    # PYTHON: __init__ runs when the exception is created: ToolError("msg", 504).
    # super().__init__(message) lets the parent Exception store the message, so
    # str(err) still returns it. We add one extra attribute, status_code, which the
    # API layer (app/main.py) uses as the HTTP status.
    # CONCEPT (HTTP): 404 = not found, 504 = an upstream service timed out.
    def __init__(self, message: str, status_code: int = 404):
        super().__init__(message)
        self.status_code = status_code


# PYTHON: `async def` defines a COROUTINE FUNCTION. Calling get_deductible("M123")
# does NOT run it; it returns a paused coroutine object. It only runs when
# something awaits it (`await get_deductible(...)`) or schedules it (asyncio.gather).
#
# PYTHON: `member_id: str` and `-> dict` are TYPE HINTS. Python doesn't enforce
# them at runtime, but editors use them for autocomplete, FastAPI/Pydantic use them
# to validate input (M2), and LLM tool definitions can be generated from them (M3).
#
# SECURITY: member_id is always the LOGGED-IN member, passed in by our code from
# the session token. The LLM never supplies it, so a prompt like
# "show me M456's deductible" can't change whose data is read.
async def get_deductible(member_id: str) -> dict:
    # CONCEPT: `await` = "pause this coroutine until the slow thing finishes, and let
    # the event loop run other work meanwhile". asyncio.sleep is a NON-BLOCKING wait,
    # standing in for network latency. (time.sleep would BLOCK the whole loop.)
    await asyncio.sleep(0.8)  # simulated API latency

    # PYTHON: dict.get(key) returns None if the key is missing, instead of raising
    # KeyError like data.MEMBERS[member_id] would.
    member = data.MEMBERS.get(member_id)
    if member is None:
        raise ToolError("Member not found.")

    # CONCEPT: the arithmetic happens HERE in code, not in the LLM. LLMs are
    # unreliable at math and may invent numbers; tools return exact facts and the
    # LLM only puts them into words.
    remaining = member["deductible_total"] - member["deductible_met"]
    return {"deductible_total": member["deductible_total"],
            "deductible_met": member["deductible_met"],
            "remaining": remaining}


async def get_claim_status(member_id: str, claim_id: str) -> dict:
    await asyncio.sleep(1.0)
    claim = data.CLAIMS.get(claim_id)

    # SECURITY: AUTHORIZATION INSIDE THE TOOL. claim_id may come from the member's
    # chat text (via the LLM), so it's untrusted. We check ownership here, at the
    # data access point, so even a prompt-injected LLM can't read another member's claim.
    #
    # SECURITY: we return the SAME message for "doesn't exist" and "belongs to
    # someone else". Different messages would let an attacker probe which claim IDs
    # exist (an "enumeration" leak).
    if claim is None or claim["member_id"] != member_id:
        # PYTHON: f-string. Anything in {braces} is evaluated and inserted into the text.
        raise ToolError(f"No claim {claim_id} found on your account.")

    # PYTHON: two features on one line:
    #   1. DICT COMPREHENSION {k: v for k, v in claim.items() if k != "member_id"}
    #      builds a copy of the claim WITHOUT the member_id field (data minimization:
    #      don't hand the LLM more than it needs).
    #   2. ** UNPACKING merges that copy into a new dict that starts with claim_id.
    return {"claim_id": claim_id, **{k: v for k, v in claim.items() if k != "member_id"}}


async def check_coverage(member_id: str, cpt_code: str) -> dict:
    await asyncio.sleep(1.5)  # the slowest tool, so it decides the total time in gather()
    member = data.MEMBERS.get(member_id)
    if member is None:
        raise ToolError("Member not found.")

    # Coverage depends on the member's PLAN, so we look up their plan_id first
    # and then use the (plan_id, cpt_code) tuple as the key.
    rule = data.COVERAGE.get((member["plan_id"], cpt_code))
    if rule is None:
        # CONCEPT: "not covered" is a normal ANSWER, not an error, so we return it
        # instead of raising. Raise only when the tool genuinely couldn't do its job.
        return {"cpt_code": cpt_code, "covered": False, "note": "Not listed in your plan."}
    return {"cpt_code": cpt_code, **rule}


# PYTHON: Coroutine[Any, Any, T] is the TYPE of an un-awaited coroutine object
# (what you get from calling an async function without awaiting it).
# `def run_tool[T](...)` is Python 3.12 GENERIC syntax: T is a placeholder type, so
# run_tool(get_deductible(...)) is understood to return whatever get_deductible
# returns (a dict). Purely for editors/type checkers; it changes nothing at runtime.
async def run_tool[T](coro: Coroutine[Any, Any, T], timeout_s: float = 2.0) -> T:
    """Run one tool call with a time limit. Every tool call in the agent goes through this.

    CONCEPT: a hung API must never freeze the chat. Real services sometimes stop
    responding without erroring; without a timeout, the member waits forever and
    the server holds that request open.
    """
    try:
        # CONCEPT: `async with asyncio.timeout(...)` is an ASYNC CONTEXT MANAGER
        # (Python 3.11+). If the code inside hasn't finished after timeout_s seconds,
        # asyncio CANCELS it (the awaited coroutine gets a CancelledError at its
        # current `await`) and then raises TimeoutError here.
        #
        # CONCEPT: this only works because the tool `await`s. Cancellation can only
        # happen at an `await` point. A tool stuck in blocking code (time.sleep,
        # requests.get) can't be cancelled, and the timeout fires late or never.
        async with asyncio.timeout(timeout_s):
            return await coro
    except TimeoutError:
        # CONCEPT: convert a low-level error into our ToolError with a member-friendly
        # message. The agent then treats it like any other expected tool failure.
        # `from None` hides the original traceback chain, since this is expected, not a bug.
        raise ToolError("That lookup is taking too long. Please try again.", status_code=504) from None


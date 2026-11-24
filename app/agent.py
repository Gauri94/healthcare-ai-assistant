"""The assistant's brain: decides which tools to run for a message, runs them, replies.

TEMPORARY VERSION (M2): a simple rule-based planner, NO LLM yet. It spots claim
IDs, CPT codes and the word "deductible" with regular expressions. In M3 an LLM
replaces the planning and the wording, but the shape stays the same:
  1. plan which tools to call
  2. run them CONCURRENTLY with timeouts (gather + run_tool)
  3. turn the results into a reply

CONCEPT (production): keeping a no-LLM path is actually useful later. If the LLM
provider is down or the kill switch / spend cap trips, the assistant can fall
back to this deterministic mode instead of failing completely.
"""

import asyncio
import re
from collections.abc import Coroutine
from typing import Any

from app.schemas import ToolCall
from app.tools import ToolError, check_coverage, get_claim_status, get_deductible, run_tool

# PYTHON: re.compile builds a reusable REGULAR EXPRESSION (a text pattern).
#   \b       = word boundary (so "C-10011" doesn't match as "C-1001")
#   C-\d{4}  = "C-" followed by exactly 4 digits
#   \d{5}    = exactly 5 digits (CPT codes are 5 characters; real ones can also
#              end in a letter, which we ignore for the MVP)
#   re.IGNORECASE so "c-1001" also matches
CLAIM_PATTERN = re.compile(r"\bC-\d{4}\b", re.IGNORECASE)
CPT_PATTERN = re.compile(r"\b\d{5}\b")

HELP_TEXT = (
    "I can help with your claims (e.g. 'status of claim C-1001'), "
    "coverage for a procedure (give me the CPT code, e.g. 73721), "
    "and how much of your deductible is left."
)


async def answer(member_id: str, message: str) -> tuple[str, list[ToolCall]]:
    """Return (reply_text, tool_call_records) for one member message."""

    # ---- 1. PLAN ----
    # Each planned call: (tool name, args to record, the coroutine to run).
    # PYTHON: creating the coroutine here does NOT run it yet (coroutines are lazy).
    planned: list[tuple[str, dict[str, str], Coroutine[Any, Any, dict]]] = []

    for claim_id in CLAIM_PATTERN.findall(message):
        claim_id = claim_id.upper()
        planned.append(("get_claim_status", {"claim_id": claim_id},
                        get_claim_status(member_id, claim_id)))

    for cpt_code in CPT_PATTERN.findall(message):
        planned.append(("check_coverage", {"cpt_code": cpt_code},
                        check_coverage(member_id, cpt_code)))

    if "deductible" in message.lower():
        planned.append(("get_deductible", {}, get_deductible(member_id)))

    if not planned:
        return HELP_TEXT, []

    # ---- 2. RUN, concurrently, each with a timeout ----
    # PYTHON: the * in gather(*[...]) UNPACKS the list into separate arguments,
    # i.e. gather(a, b, c) instead of gather([a, b, c]).
    # return_exceptions=True: one failed tool doesn't sink the others (see M1).
    results = await asyncio.gather(
        *(run_tool(coro) for _name, _args, coro in planned),
        return_exceptions=True,
    )

    # ---- 3. RECORD each outcome ----
    calls: list[ToolCall] = []
    # PYTHON: zip pairs up items from two lists: (planned[0], results[0]), ...
    # gather keeps results in the same order as the input, so the pairs line up.
    for (name, args, _coro), result in zip(planned, results):
        if isinstance(result, ToolError):
            calls.append(ToolCall(name=name, args=args, ok=False, error=str(result)))
        elif isinstance(result, BaseException):
            # A real bug, not an expected tool failure. Re-raise so it becomes a
            # 500 error and shows up loudly in logs, rather than being hidden.
            #
            # PYTHON: why BaseException and not Exception? gather(return_exceptions=True)
            # can return ANY BaseException, including asyncio.CancelledError (raised
            # when a request is cancelled, e.g. the member closes the tab), which is
            # NOT a subclass of Exception. Checking BaseException covers every case, so:
            #   1. CancelledError gets re-raised and cancellation keeps working, and
            #   2. the type checker (Pylance) can prove that in the `else` branch,
            #      `result` must be a dict. That was the red squiggle on the line below.
            raise result
        else:
            calls.append(ToolCall(name=name, args=args, ok=True, result=result))

    # ---- 4. REPLY ----
    reply = " ".join(_describe(call) for call in calls)
    return reply, calls


def _describe(call: ToolCall) -> str:
    """Template one tool outcome into a sentence. (M3: the LLM does this part.)

    PYTHON: the leading underscore in _describe is a convention meaning
    "private helper, not meant to be imported from other modules".
    """
    if not call.ok:
        return call.error or "Something went wrong with that lookup."

    r = call.result or {}
    if call.name == "get_deductible":
        return (f"You've met ${r['deductible_met']:,.2f} of your ${r['deductible_total']:,.2f} "
                f"deductible, so ${r['remaining']:,.2f} is left.")
    if call.name == "get_claim_status":
        owes = "not decided yet" if r["member_owes"] is None else f"${r['member_owes']:,.2f}"
        return f"Claim {r['claim_id']} ({r['service']}) is {r['status']}. You owe: {owes}."
    if call.name == "check_coverage":
        if not r["covered"]:
            return f"Procedure {r['cpt_code']} is not covered by your plan."
        parts = [f"Procedure {r['cpt_code']} is covered"]
        if r.get("prior_auth"):
            parts.append("it needs prior authorization first")
        if r.get("coinsurance") is not None:
            parts.append(f"you pay {r['coinsurance']:.0%} coinsurance after your deductible")
        if r.get("copay") is not None:
            parts.append(f"with a ${r['copay']:,.2f} copay")
        return ", ".join(parts) + "."
    return str(r)

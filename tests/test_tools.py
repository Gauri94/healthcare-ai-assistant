"""UNIT tests: the tools and the validation of LLM tool calls, in isolation.

CONCEPT (test pyramid): many small, fast UNIT tests at the bottom (one function
at a time), fewer INTEGRATION tests above (several parts together: API + agent
+ tools), and a few end-to-end checks or evals at the top (real LLM). Unit tests
pinpoint exactly what broke; integration tests catch parts that don't fit together.

CONCEPT: async tests are plain `async def test_...` functions. pytest-asyncio runs
them on an event loop (asyncio_mode = "auto" in pyproject.toml).
"""

import asyncio

import pytest

from app.tool_specs import parse_tool_call
from app.tools import ToolError, check_coverage, get_claim_status, get_deductible, run_tool


# ---- Tools: facts and arithmetic ----

async def test_deductible_remaining_is_computed_in_code():
    result = await get_deductible("M123")
    assert result == {"deductible_total": 2000.0, "deductible_met": 1150.0, "remaining": 850.0}


async def test_coverage_for_listed_procedure():
    result = await check_coverage("M123", "73721")
    assert result["covered"] is True
    assert result["prior_auth"] is True
    assert result["coinsurance"] == 0.20


async def test_unlisted_procedure_is_not_covered_rather_than_an_error():
    result = await check_coverage("M123", "99999")
    assert result["covered"] is False


async def test_claim_result_does_not_expose_member_id():
    # Data minimization: the LLM doesn't need the internal member ID.
    result = await get_claim_status("M123", "C-1001")
    assert "member_id" not in result
    assert result["claim_id"] == "C-1001"


# ---- Tools: authorization inside the tool ----

async def test_member_cannot_read_another_members_claim():
    # C-2001 belongs to M456.
    with pytest.raises(ToolError, match="No claim C-2001 found"):
        await get_claim_status("M123", "C-2001")


async def test_other_members_claim_and_missing_claim_give_identical_errors():
    # SECURITY: identical messages, so an attacker can't tell which claim IDs exist.
    # PYTHON: pytest.raises(...) as exc_info captures the exception for inspection.
    with pytest.raises(ToolError) as someone_elses:
        await get_claim_status("M123", "C-2001")
    with pytest.raises(ToolError) as nonexistent:
        await get_claim_status("M123", "C-9999")
    assert str(someone_elses.value).replace("C-2001", "X") == str(nonexistent.value).replace("C-9999", "X")
    assert someone_elses.value.status_code == nonexistent.value.status_code == 404


# ---- run_tool: timeouts ----

async def test_run_tool_times_out_with_friendly_error():
    async def hung_api() -> dict:
        await asyncio.sleep(10)
        return {}

    with pytest.raises(ToolError) as exc_info:
        await run_tool(hung_api(), timeout_s=0.05)
    assert exc_info.value.status_code == 504
    assert "taking too long" in str(exc_info.value)


async def test_run_tool_returns_result_as_soon_as_ready():
    async def quick() -> dict:
        return {"ok": True}

    assert await run_tool(quick(), timeout_s=1) == {"ok": True}


# ---- parse_tool_call: treating LLM output as untrusted input ----

def test_unknown_tool_is_refused():
    with pytest.raises(ToolError, match="Unknown tool"):
        parse_tool_call("delete_all_claims", "{}")


def test_invalid_json_arguments_are_refused():
    with pytest.raises(ToolError, match="not valid JSON"):
        parse_tool_call("get_claim_status", "{claim_id: C-1001")  # broken JSON


def test_non_object_arguments_are_refused():
    with pytest.raises(ToolError, match="JSON object"):
        parse_tool_call("get_claim_status", '["C-1001"]')


def test_smuggled_member_id_argument_is_dropped():
    # SECURITY: even if the model adds member_id, it never reaches the tool.
    args = parse_tool_call("get_claim_status", '{"claim_id": "C-1001", "member_id": "M456"}')
    assert args == {"claim_id": "C-1001"}


def test_missing_required_argument_is_refused():
    with pytest.raises(ToolError, match="missing: claim_id"):
        parse_tool_call("get_claim_status", "{}")


# CONCEPT (PARAMETRIZE): one test function, run once per input row. Good for
# covering many cases without copy-pasting the test.
@pytest.mark.parametrize("raw", ["", "{}", "null"])
def test_tool_without_parameters_accepts_empty_arguments(raw):
    # "null" parses to None, which isn't a dict, so it must be refused; "" and "{}" are fine.
    if raw == "null":
        with pytest.raises(ToolError):
            parse_tool_call("get_deductible", raw)
    else:
        assert parse_tool_call("get_deductible", raw) == {}

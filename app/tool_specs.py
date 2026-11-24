"""Tool definitions shown to the LLM, and the code that runs a tool the LLM picked.

CONCEPT (tool calling): the LLM never runs code. We send it a list of tool
DEFINITIONS (name, description, JSON Schema for the arguments). If it decides a
tool would help, it replies with a TOOL CALL instead of text, e.g.
    {"name": "get_claim_status", "arguments": "{\"claim_id\": \"C-1001\"}"}
OUR code runs the real function and sends the result back. The LLM then either
calls more tools or writes the final answer.

CONCEPT: the DESCRIPTIONS matter a lot. They are effectively prompts: the model
decides which tool to use, and when, mostly from them.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from app.tools import ToolError, check_coverage, get_claim_status, get_deductible

# OpenAI-style tool format (used by OpenRouter and most providers).
# SECURITY: NONE of these schemas has a member_id parameter. The LLM cannot even
# express "look up member M456". Our code injects the logged-in member's ID when
# it runs the tool (see run_llm_tool_call below).
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_deductible",
            "description": "Get the logged-in member's annual deductible: total, amount met so far "
                           "this year, and amount remaining.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_claim_status",
            "description": "Look up one of the logged-in member's claims by claim ID: service, status "
                           "(paid, pending, denied), amount billed and what the member owes.",
            "parameters": {
                "type": "object",
                "properties": {
                    # JSON Schema "pattern" documents the format for the model. We still
                    # validate in code, because the model can ignore the schema.
                    "claim_id": {"type": "string", "pattern": "^C-\\d{4}$",
                                 "description": "Claim ID like C-1001."},
                },
                "required": ["claim_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_coverage",
            "description": "Check whether the logged-in member's plan covers a procedure, given its "
                           "5-digit CPT code. Returns covered (true/false), whether prior "
                           "authorization is needed, and coinsurance or copay. Only call this with "
                           "a CPT code the member provided or explicitly confirmed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cpt_code": {"type": "string", "pattern": "^\\d{5}$",
                                 "description": "5-digit CPT procedure code, e.g. 73721."},
                },
                "required": ["cpt_code"],
            },
        },
    },
]

# PYTHON: Callable[..., Awaitable[dict]] = "any function that, when called, returns
# something you can await that gives a dict", i.e. our async tools.
# This ALLOWLIST maps names the LLM may use to real functions. A name that isn't
# here is refused, even if the model invents one.
_TOOL_FUNCTIONS: dict[str, Callable[..., Awaitable[dict]]] = {
    "get_deductible": get_deductible,
    "get_claim_status": get_claim_status,
    "check_coverage": check_coverage,
}


def parse_tool_call(name: str, raw_arguments: str) -> dict[str, str]:
    """Validate a tool call from the LLM BEFORE running anything. Raises ToolError if invalid.

    CONCEPT: treat LLM output as UNTRUSTED INPUT, like data from a web form. Models
    sometimes produce invalid JSON, call tools that don't exist, or add extra arguments.
    """
    if name not in _TOOL_FUNCTIONS:
        raise ToolError(f"Unknown tool: {name}", status_code=400)
    try:
        args = json.loads(raw_arguments or "{}")  # PYTHON: json.loads parses a JSON string into Python
    except json.JSONDecodeError:
        raise ToolError("The tool arguments were not valid JSON.", status_code=400) from None
    if not isinstance(args, dict):
        raise ToolError("The tool arguments must be a JSON object.", status_code=400)

    # Keep ONLY the parameters this tool declares. Anything else is dropped, so a
    # model that adds {"member_id": "M456"} achieves nothing.
    params = next(t for t in TOOL_SPECS if t["function"]["name"] == name)["function"]["parameters"]
    cleaned = {k: str(v) for k, v in args.items() if k in params["properties"]}

    # The model can also FORGET a required argument (e.g. claim_id).
    missing = [p for p in params["required"] if p not in cleaned]
    if missing:
        raise ToolError(f"{name} is missing: {', '.join(missing)}.", status_code=400)
    return cleaned


async def run_llm_tool_call(member_id: str, name: str, args: dict[str, str]) -> dict:
    """Run a validated tool call for the logged-in member."""
    func = _TOOL_FUNCTIONS[name]
    # PYTHON: **args unpacks the dict into keyword arguments:
    #   func(member_id, **{"claim_id": "C-1001"})  ==  func(member_id, claim_id="C-1001")
    # SECURITY: member_id comes from the login token (passed in by our code), never from args.
    return await func(member_id, **args)

"""The same agent, built with LangGraph, plus MULTI-TURN MEMORY per conversation.

Select it with AGENT_ENGINE=langgraph in .env (the default). AGENT_ENGINE=loop
uses the hand-written loop in app/llm_agent.py. Both use the same tools,
validation, timeouts, spend cap and fallback.

CONCEPT (LangGraph in one paragraph): you describe your agent as a GRAPH.
  - STATE: a typed dict that flows through the graph (messages, counters...).
  - NODES: plain async functions. Each receives the current state and returns
    ONLY the fields it wants to change (a partial update).
  - EDGES: which node runs next. CONDITIONAL EDGES call a small "router"
    function that looks at the state and picks the next node.
  - REDUCERS: how a node's update is merged into the state. Default = replace
    the value; with operator.add, lists are APPENDED to instead.
  - CHECKPOINTER: saves the state after every step, per THREAD (conversation).
    The next request on the same thread starts from the saved state, which is
    what gives the assistant memory of earlier turns.

Our graph:

    START -> call_model --(tool calls?)--> run_tools --(steps left?)--> call_model ...
                 |                              |
                 +--(text answer / fallback)--> END     +--(step limit)--> step_limit -> END

Print it as a diagram:   uv run python -m scripts.show_graph
"""

import logging
import operator
import uuid
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

import openai
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from openai.types.chat import ChatCompletionMessageFunctionToolCall

from app.budget import tracker
from app.config import get_settings
from app.llm_agent import SYSTEM_PROMPT, AgentResult, _client, _fallback, execute_tool_calls
from app.schemas import ToolCall
from app.tool_specs import TOOL_SPECS

logger = logging.getLogger(__name__)

STEP_LIMIT_REPLY = ("Sorry, I couldn't finish answering that. Could you rephrase it, "
                    "or ask one thing at a time?")


# =============================================================================
# 1. STATE: everything the graph knows, flowing from node to node
# =============================================================================
# PYTHON: TypedDict describes a dict with fixed, typed keys. LangGraph reads these
# type hints to know the state's shape.
class AgentState(TypedDict):
    # CONCEPT (REDUCER): Annotated[list, operator.add] tells LangGraph to APPEND a
    # node's returned messages to the existing list (old + new), not replace it.
    # Together with the checkpointer, this list is the conversation history across
    # turns: earlier questions, tool calls, tool results and answers.
    messages: Annotated[list[dict], operator.add]

    # The fields below have NO reducer, so an update REPLACES the value.
    # We reset them at the start of every turn (see answer() below).
    member_id: str              # SECURITY: set by OUR code from the login token, every turn
    steps: int                  # LLM calls made in this turn (for the step limit)
    turn_tool_calls: list[dict]  # tool records for this turn's API response
    turn_cost: float            # LLM cost of this turn
    fallback_reason: str | None  # set if we must fall back to the rule-based planner
    reply: str                  # the final answer text for this turn


# =============================================================================
# 2. NODES: async functions that take the state and return a PARTIAL update
# =============================================================================

def _recent_history(messages: list[dict], max_messages: int = 20) -> list[dict]:
    """Keep only the latest messages to send to the LLM.

    CONCEPT (context engineering): history grows every turn, and every message is
    resent (and paid for) on every call. So we cap it. The cut must start at a
    "user" message: starting at a "tool" message whose tool call was cut off
    would be an invalid conversation for the LLM API.
    (A better production approach: SUMMARIZE older turns instead of dropping them.)
    """
    recent = messages[-max_messages:]
    while recent and recent[0].get("role") != "user":
        recent = recent[1:]
    return recent


async def call_model(state: AgentState) -> dict:
    """NODE: ask the LLM what to do next, offering the tools."""
    s = get_settings()

    # Guardrail: spend cap / kill switch, checked before EVERY LLM call.
    if not tracker.can_spend():
        return {"fallback_reason": "daily LLM budget reached"}

    try:
        response = await _client().chat.completions.create(
            model=s.llm_model,
            # The system prompt is added at call time and NOT stored in the state,
            # so updating the prompt also applies to old conversations.
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *_recent_history(state["messages"])],  # type: ignore[list-item]
            tools=TOOL_SPECS,  # type: ignore[arg-type]
            tool_choice="auto",
            temperature=0,
            max_tokens=s.llm_max_tokens,
        )
    except (openai.APIError, TimeoutError) as e:
        return {"fallback_reason": f"LLM error: {type(e).__name__}"}

    cost = 0.0
    if response.usage:
        cost = tracker.record(response.usage.prompt_tokens, response.usage.completion_tokens)
    msg = response.choices[0].message

    # CONCEPT: return ONLY what changed. "messages" has the operator.add reducer, so
    # this one message is APPENDED to the history. The other keys are REPLACED.
    update: dict = {
        "messages": [msg.model_dump(exclude_none=True)],
        "steps": state["steps"] + 1,
        "turn_cost": state["turn_cost"] + cost,
    }
    if not msg.tool_calls:
        update["reply"] = (msg.content or "").strip() or "Sorry, I couldn't come up with an answer."
    return update


async def run_tools(state: AgentState) -> dict:
    """NODE: run the tools the LLM asked for (concurrently, validated, with timeouts)."""
    last = state["messages"][-1]
    # The checkpointer stores plain dicts, so we rebuild the SDK's typed tool-call
    # objects that execute_tool_calls() expects. model_validate also checks the shape.
    tool_calls = [ChatCompletionMessageFunctionToolCall.model_validate(tc)
                  for tc in last.get("tool_calls", []) if tc.get("type") == "function"]

    tool_messages, records = await execute_tool_calls(state["member_id"], tool_calls)
    return {
        "messages": tool_messages,  # appended (reducer)
        # Stored as plain dicts so the checkpointer can save them easily.
        "turn_tool_calls": state["turn_tool_calls"] + [r.model_dump() for r in records],
    }


async def step_limit(state: AgentState) -> dict:
    """NODE: too many LLM calls this turn. Stop with a polite message."""
    logger.warning("LangGraph agent hit max_agent_steps=%d", get_settings().max_agent_steps)
    # We also add the reply to the history, so the next turn sees a complete exchange.
    return {"reply": STEP_LIMIT_REPLY,
            "messages": [{"role": "assistant", "content": STEP_LIMIT_REPLY}]}


# =============================================================================
# 3. ROUTERS for conditional edges: look at the state, return the next node's name
# =============================================================================
# PYTHON: Literal["a", "b"] means "only these exact strings". LangGraph uses the
# return type hint to know which nodes this router can lead to (for the diagram).

# "__end__" is the value of LangGraph's END constant. We return the literal string
# because END is typed as a plain str, which the type checker can't match to Literal.
def after_model(state: AgentState) -> Literal["run_tools", "__end__"]:
    if state["fallback_reason"]:
        return "__end__"  # answer() will run the rule-based fallback
    if state["messages"][-1].get("tool_calls"):
        return "run_tools"
    return "__end__"  # a plain text answer: done


def after_tools(state: AgentState) -> Literal["call_model", "step_limit"]:
    if state["steps"] >= get_settings().max_agent_steps:
        return "step_limit"
    return "call_model"  # let the LLM read the tool results


# =============================================================================
# 4. BUILD AND COMPILE THE GRAPH
# =============================================================================
@lru_cache  # build once, reuse for every request (see app/config.py for lru_cache)
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("call_model", call_model)
    builder.add_node("run_tools", run_tools)
    builder.add_node("step_limit", step_limit)

    builder.add_edge(START, "call_model")                    # always start by asking the LLM
    builder.add_conditional_edges("call_model", after_model)  # tools, or done?
    builder.add_conditional_edges("run_tools", after_tools)   # back to the LLM, or stop?
    builder.add_edge("step_limit", END)

    # CONCEPT (CHECKPOINTER): saves the state after every node, keyed by thread_id.
    # InMemorySaver keeps it in this process's memory: lost on restart, and not
    # shared between servers. Production swaps in a persistent one (e.g. the
    # Postgres or Redis checkpointer packages) with no other code changes.
    # Checkpoints also enable time travel (replay from an earlier step) and
    # human-in-the-loop pauses (interrupts), which we may use for human handoff.
    return builder.compile(checkpointer=InMemorySaver())


# =============================================================================
# 5. ENTRY POINT used by the API
# =============================================================================
async def answer(member_id: str, message: str, conversation_id: str | None = None) -> AgentResult:
    s = get_settings()
    conversation_id = conversation_id or str(uuid.uuid4())  # new conversation if none given
    if s.openrouter_api_key is None:
        result = await _fallback(member_id, message, "no API key configured")
        result.conversation_id = conversation_id
        return result

    graph = build_graph()
    # CONCEPT (THREADS): the checkpointer keys saved state by thread_id.
    # SECURITY: we put the member_id INSIDE the thread_id. If member M123 sends
    # M456's conversation_id, they get "M123:<id>", a different (empty) thread,
    # never M456's history. The member can choose the conversation, never the owner.
    config = {
        "configurable": {"thread_id": f"{member_id}:{conversation_id}"},
        # CONCEPT: recursion_limit = max node runs per invocation. LangGraph's own
        # backstop against infinite loops, on top of our max_agent_steps check.
        "recursion_limit": s.max_agent_steps * 2 + 5,
    }

    # The input for this turn: the new user message is APPENDED to the saved history
    # (reducer), and the per-turn fields are RESET (no reducer, so replaced).
    turn_input: AgentState = {
        "messages": [{"role": "user", "content": message}],
        "member_id": member_id,
        "steps": 0,
        "turn_tool_calls": [],
        "turn_cost": 0.0,
        "fallback_reason": None,
        "reply": "",
    }
    # CONCEPT: ainvoke runs the graph from START until it reaches END, then returns
    # the final state. (astream would yield each step as it happens: useful for M4.)
    final = await graph.ainvoke(turn_input, config)  # type: ignore[arg-type]

    if final["fallback_reason"]:
        result = await _fallback(member_id, message, final["fallback_reason"])
        result.cost_usd = final["turn_cost"]
        # CONCEPT: update_state writes into a thread's saved state from OUTSIDE the
        # graph. We record the fallback's reply so the history stays a complete
        # user -> assistant exchange for the next turn.
        await graph.aupdate_state(config, {"messages": [{"role": "assistant", "content": result.reply}]})  # type: ignore[arg-type]
    else:
        result = AgentResult(
            reply=final["reply"],
            tool_calls=[ToolCall(**tc) for tc in final["turn_tool_calls"]],
            llm_calls=final["steps"],
            cost_usd=final["turn_cost"],
        )
    result.conversation_id = conversation_id
    return result

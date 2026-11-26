"""Shared test setup (fixtures). pytest loads this file automatically.

CONCEPT (pytest FIXTURES): a fixture is a function that prepares something a
test needs. A test asks for it by NAME as a parameter:
    def test_x(fake_llm): ...
pytest runs the fixture first and passes in its result. Code after `yield` in a
fixture runs AFTER the test (cleanup). autouse=True fixtures run for every test
without being asked for.

CONCEPT (MOCKING the LLM): tests never call the real LLM. A real call costs
money, is slow, needs network, and gives slightly different answers each time,
so tests would be flaky. Instead, FakeLLM returns SCRIPTED replies we choose.
That lets us force exact situations: "the model asks for someone else's claim",
"the model sends invalid JSON", "the model loops forever", and check that OUR
code handles each one. We're testing our guardrails, not the model.
(Testing the real model's quality is a separate job: evals, in M10.)

CONCEPT (monkeypatch): pytest's built-in fixture for temporarily replacing
attributes or environment variables during one test. Everything is restored
automatically afterwards, so tests can't leak changes into each other.
"""

import copy
import json
from collections.abc import Callable
from functools import lru_cache

import pytest
from openai.types.chat import ChatCompletion

from app import budget, config, graph_agent, llm_agent, tools

ASHA = {"Authorization": "Bearer token-asha"}  # member M123
BEN = {"Authorization": "Bearer token-ben"}    # member M456


# =============================================================================
# Building fake LLM responses
# =============================================================================
# The real SDK returns ChatCompletion objects. We build the same type from a dict,
# so our code can't tell the difference.

def _completion(message: dict, finish_reason: str) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "fake", "object": "chat.completion", "created": 0, "model": "fake-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        # Fixed token counts, so cost assertions are predictable.
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
    })


def text_reply(content: str) -> ChatCompletion:
    """The model answers in plain text (no tools)."""
    return _completion({"role": "assistant", "content": content}, "stop")


def tool_reply(*calls: tuple[str, dict | str]) -> ChatCompletion:
    """The model asks for one or more tools. Each call is (tool_name, arguments).
    Pass arguments as a dict (normal) or a raw string (to test invalid JSON)."""
    tool_calls = [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
        for i, (name, args) in enumerate(calls)
    ]
    return _completion({"role": "assistant", "content": None, "tool_calls": tool_calls}, "tool_calls")


# =============================================================================
# The fake client
# =============================================================================
class FakeLLM:
    """Stands in for AsyncOpenAI. Mimics only the part we use:
    `await client.chat.completions.create(...)` and `await client.close()`."""

    def __init__(self) -> None:
        self.script: list[ChatCompletion | Exception] = []  # replies, returned in order
        self.default: Callable[[], ChatCompletion] | None = None  # used once the script runs out
        self.requests: list[list[dict]] = []  # the messages the agent sent on each call
        # PYTHON: a tiny trick so `fake.chat.completions.create` works: both
        # attributes just point back at this same object.
        self.chat = self
        self.completions = self

    async def create(self, *, messages, **kwargs) -> ChatCompletion:
        # deepcopy: record exactly what was sent, even if the agent mutates the list later.
        self.requests.append(copy.deepcopy(list(messages)))
        if self.script:
            reply = self.script.pop(0)
        elif self.default:
            reply = self.default()
        else:
            raise AssertionError("FakeLLM called more times than the test scripted")
        if isinstance(reply, Exception):
            raise reply  # simulate a provider error
        return reply

    async def close(self) -> None:
        pass

    @property
    def calls(self) -> int:
        return len(self.requests)

    def tool_results_sent(self, call_index: int) -> list[dict]:
        """The tool results our code sent back to the LLM, as parsed dicts."""
        return [json.loads(m["content"]) for m in self.requests[call_index] if m["role"] == "tool"]


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(autouse=True)
def isolated_app(monkeypatch):
    """Runs for EVERY test: a clean, offline, fast app."""
    # A fake key, set in the environment so it OVERRIDES the real key in .env.
    # Agent tests also replace the client, so no request can ever reach OpenRouter.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setenv("DAILY_LLM_BUDGET_USD", "1.0")
    config.get_settings.cache_clear()  # re-read settings with the values above

    # No simulated tool latency: tests run in milliseconds.
    async def no_latency(seconds: float) -> None:
        return None
    monkeypatch.setattr(tools, "simulate_latency", no_latency)

    # Fresh spend total and fresh LangGraph memory for every test.
    monkeypatch.setattr(budget.tracker, "_spent_usd", 0.0)
    graph_agent.build_graph.cache_clear()

    yield  # ---- the test runs here ----

    config.get_settings.cache_clear()
    graph_agent.build_graph.cache_clear()


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    """Replace the real LLM client with a FakeLLM, in BOTH agent engines."""
    fake = FakeLLM()
    # Wrapped in lru_cache so it behaves like the real _client (close_client() uses
    # cache_info and cache_clear). graph_agent imported _client by name, so it
    # must be patched there too.
    factory = lru_cache(lambda: fake)

    # PYTHON: monkeypatch.setattr(target, "name", new_value) has 3 parameters:
    #   1. target    = the OBJECT to change (here a module: app.llm_agent)
    #   2. "name"    = the attribute on that object to replace, as a STRING
    #   3. new_value = what to put there instead (our fake factory)
    # It's like writing `llm_agent._client = factory`, except pytest remembers the
    # original value and puts it back automatically when the test ends.
    #
    # Why patch BOTH modules? graph_agent.py did `from app.llm_agent import _client`.
    # That import copied a REFERENCE to the function into graph_agent's own namespace,
    # creating a second name, graph_agent._client, that points at the same function.
    # Replacing llm_agent._client rebinds only THAT name; graph_agent._client would
    # still point at the real client. So each module's name must be patched where
    # it's USED. (Rule of thumb: "patch where it's looked up, not where it's defined.")
    monkeypatch.setattr(llm_agent, "_client", factory)
    monkeypatch.setattr(graph_agent, "_client", factory)
    return fake


# CONCEPT (PARAMETRIZED fixture): every test that uses `engine` runs TWICE, once
# per value. Our security and guardrail tests therefore prove that BOTH agent
# implementations behave the same way.
@pytest.fixture(params=["loop", "langgraph"])
def engine(request, monkeypatch) -> str:
    monkeypatch.setenv("AGENT_ENGINE", request.param)
    config.get_settings.cache_clear()
    return request.param


async def ask(engine: str, member_id: str, message: str, conversation_id: str | None = None):
    """Call whichever agent engine is selected."""
    if engine == "langgraph":
        return await graph_agent.answer(member_id, message, conversation_id)
    return await llm_agent.answer(member_id, message)

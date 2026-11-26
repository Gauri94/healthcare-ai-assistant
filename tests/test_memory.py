"""LangGraph multi-turn memory: history is remembered per conversation and per member."""

from app.graph_agent import _recent_history, answer
from tests.conftest import text_reply


async def test_second_turn_includes_the_first_turn(fake_llm):
    fake_llm.script = [text_reply("Yes, 73721 is covered."), text_reply("You asked about 73721.")]

    first = await answer("M123", "Is 73721 covered?")
    await answer("M123", "What did I ask first?", first.conversation_id)

    second_request = fake_llm.requests[1]
    contents = [m.get("content") for m in second_request]
    assert "Is 73721 covered?" in contents        # earlier question
    assert "Yes, 73721 is covered." in contents   # earlier answer
    assert contents[-1] == "What did I ask first?"


async def test_new_conversation_starts_empty(fake_llm):
    fake_llm.script = [text_reply("a"), text_reply("b")]
    await answer("M123", "Is 73721 covered?")
    await answer("M123", "hello")  # no conversation_id -> a new conversation

    assert "Is 73721 covered?" not in str(fake_llm.requests[1])


async def test_other_member_cannot_read_a_conversation_by_reusing_its_id(fake_llm):
    # SECURITY: the thread is keyed by member_id + conversation_id.
    fake_llm.script = [text_reply("Your knee MRI is covered."), text_reply("This is your first message.")]
    asha = await answer("M123", "Is my knee MRI 73721 covered?")
    await answer("M456", "What did I ask first?", asha.conversation_id)  # Ben reuses Asha's ID

    bens_request = str(fake_llm.requests[1])
    assert "knee MRI" not in bens_request
    assert "73721" not in bens_request


async def test_fallback_reply_is_saved_so_history_stays_complete(fake_llm, monkeypatch):
    from app import budget
    fake_llm.script = [text_reply("ok")]
    monkeypatch.setattr(budget.tracker, "_spent_usd", 1.0)  # budget reached: turn 1 falls back
    first = await answer("M123", "deductible?")
    monkeypatch.setattr(budget.tracker, "_spent_usd", 0.0)  # budget available again
    await answer("M123", "thanks", first.conversation_id)

    roles = [m["role"] for m in fake_llm.requests[0]]
    assert roles == ["system", "user", "assistant", "user"]  # the fallback answer was recorded


def test_history_trimming_never_starts_with_an_orphaned_tool_message():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    trimmed = _recent_history(history, max_messages=4)  # a naive cut would start at the tool message
    assert trimmed[0]["role"] == "user"
    assert trimmed == history[4:]

"""M6 safety tests: emergency messages must get the fixed reply and NEVER reach the LLM.

SDET framing: a safety rule has two ways to fail, so we test both directions:
  - MUST FLAG:   real emergencies, in many phrasings  (protects RECALL)
  - MUST NOT:    look-alike benefits questions         (protects PRECISION)
The second list matters: a rule that fires on "is stroke rehab covered?" would
make the assistant useless for exactly the members who need it most.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.safety import EmergencyCategory, check_emergency
from tests.conftest import ASHA, text_reply

MEDICAL, SELF_HARM = EmergencyCategory.MEDICAL, EmergencyCategory.SELF_HARM


# =============================================================================
# Unit tests: the rules on their own
# =============================================================================

@pytest.mark.parametrize("message, category, rule", [
    ("I have crushing chest pain", MEDICAL, "chest_pain"),
    ("there's a lot of pressure... pain in my chest", MEDICAL, "chest_pain"),
    ("I think my husband is having a heart attack", MEDICAL, "heart_attack"),
    ("I can't breathe", MEDICAL, "cant_breathe"),
    ("I can’t breathe", MEDICAL, "cant_breathe"),          # curly apostrophe (phone keyboard)
    ("my son is struggling to breathe", MEDICAL, "cant_breathe"),
    ("I think my mom is having a stroke", MEDICAL, "stroke"),
    ("her face is drooping and she has slurred speech", MEDICAL, "stroke"),
    ("my dad passed out and won't wake up", MEDICAL, "unconscious"),
    ("bleeding that won't stop", MEDICAL, "severe_bleeding"),
    ("my friend overdosed", MEDICAL, "overdose"),
    ("my throat is closing after eating peanuts", MEDICAL, "anaphylaxis"),
    ("my daughter is having a seizure", MEDICAL, "seizure"),
    ("I want to kill myself", SELF_HARM, "self_harm"),
    ("I've been feeling suicidal", SELF_HARM, "self_harm"),
    ("honestly I'd be better off dead", SELF_HARM, "self_harm"),
    ("I DON'T WANT TO LIVE ANYMORE", SELF_HARM, "self_harm"),  # case doesn't matter
])
def test_emergencies_are_flagged(message, category, rule):
    result = check_emergency(message)
    assert result is not None, f"MISSED EMERGENCY: {message!r}"
    assert (result.category, result.rule) == (category, rule)


def test_self_harm_wins_when_both_match():
    # Matches both "overdose" (medical) and "want to die" (self-harm). The 988 reply
    # is the right one: it also tells them to call 911 if in immediate danger.
    result = check_emergency("I overdosed on purpose, I want to die")
    assert result is not None and result.category == SELF_HARM
    assert "988" in result.reply and "911" in result.reply


@pytest.mark.parametrize("message", [
    "Is stroke rehab covered?",
    "My dad had a stroke last year, is physical therapy covered?",
    "I had a heart attack in 2024, does my plan cover cardiac rehab?",
    "Is my seizure medication covered?",
    "Is a chest x-ray covered?",
    "What's the copay for a breathing treatment?",
    "Can you check claim C-1001?",
    "How much of my deductible is left?",
    "Do you cover mental health therapy?",
])
def test_lookalike_benefits_questions_are_not_flagged(message):
    assert check_emergency(message) is None


# CONCEPT (xfail = "expected to fail"): this documents a KNOWN GAP instead of
# hiding it. These are real emergencies our rules miss because none of the phrases
# appear. strict=True means: the day the classifier (M6 step 2) catches them, this
# test "unexpectedly passes" and pytest reports it as a FAILURE, reminding us to
# remove the marker and turn it into a normal must-flag test.
# PYTEST: xfail is BUILT IN (no plugin). In the report: x = failed as expected,
# X / XPASS = passed unexpectedly. Unlike skip, xfail still RUNS the test.
@pytest.mark.xfail(strict=True, reason="rules miss paraphrases; classifier comes in M6 step 2")
@pytest.mark.parametrize("message", [
    "my dad's lips are turning blue",
    "she fell and hit her head and now she's confused and throwing up",
])
def test_known_gaps_rules_miss(message):
    assert check_emergency(message) is not None


# =============================================================================
# API tests: the safety check really runs BEFORE the agent
# =============================================================================

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("message, category, number", [
    ("I have chest pain and my left arm is numb", "medical", "911"),
    ("I want to end my life", "self_harm", "988"),
])
def test_chat_returns_fixed_safety_reply_without_calling_llm(client, engine, fake_llm, message, category, number):
    response = client.post("/chat", headers=ASHA, json={"message": message})
    body = response.json()

    assert response.status_code == 200
    assert body["mode"] == "emergency"
    assert body["emergency"] == category
    assert number in body["reply"]
    assert body["tool_calls"] == []
    assert body["cost_usd"] == 0.0
    # The key assertion: the LLM never saw the message (no cost, no delay, no
    # chance of a badly worded or injected reply).
    assert fake_llm.calls == 0


def test_normal_message_still_reaches_the_agent(client, engine, fake_llm):
    fake_llm.script = [text_reply("Hi! How can I help?")]
    body = client.post("/chat", headers=ASHA, json={"message": "hello"}).json()

    assert body["mode"] == "llm"
    assert body["emergency"] is None
    assert fake_llm.calls == 1

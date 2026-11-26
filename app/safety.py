"""Emergency detection: runs BEFORE the LLM on every chat message.

CONCEPT (safety layer): some messages must never get a normal assistant answer:
"I have crushing chest pain", "I want to kill myself". For these, the right reply
is fixed and urgent ("call 911 / 988"), so:
  - it's checked in plain CODE, not left to the LLM's judgment (the LLM could be
    down, over budget, prompt-injected, or just word it badly),
  - the reply is a FIXED string reviewed in advance, never generated,
  - it runs FIRST, so an emergency never waits on a slow LLM call.
This is the README's design rule: "plain code handles facts, identity and safety."

CONCEPT (recall vs precision, the key M6 idea):
  - RECALL    = of all REAL emergencies, how many did we catch?
  - PRECISION = of all messages we FLAGGED, how many were real emergencies?
For a safety check the two mistakes are NOT equally bad:
  - a FALSE NEGATIVE (missing a real emergency) can cost a life,
  - a FALSE POSITIVE (flagging "I don't have chest pain, just asking") costs the
    member one extra message.
So we deliberately favour RECALL and accept some false positives.

MEMORY TRICK: read "False Positive" as two parts. The 2nd word is what the detector
SAID (Positive = "emergency!"), the 1st word is whether it was RIGHT (False = wrong).
  - False Positive = FALSE ALARM: smoke alarm goes off for burnt toast.  Annoying.
  - False Negative = a MISS:      smoke alarm stays silent during a fire. Dangerous.
More false negatives -> lower RECALL.   More false positives -> lower PRECISION.

CONCEPT (why rules first, classifier later): rules are free, instant, predictable
and easy to test, but they only catch phrasings someone thought of. "My dad's lips
are turning blue" contains none of our phrases. Step 2 of M6 adds an ML classifier
for those. Final decision = rules OR classifier: either one can flag a message.

DESIGN CHOICE: because the classifier can only ADD flags (OR), it can't fix a rule
that fires too often. So rules here are fairly SPECIFIC phrases ("having a stroke",
not just "stroke", which would also fire on "is stroke rehab covered?"). Fuzzy cases
are the classifier's job.
"""

import re
from dataclasses import dataclass
from enum import StrEnum


# PYTHON: StrEnum (3.11+) is an Enum whose members ARE strings, so
# EmergencyCategory.MEDICAL == "medical" is True and it serializes to JSON as
# "medical". A fixed set of allowed values, with typo protection in code.
class EmergencyCategory(StrEnum):
    MEDICAL = "medical"      # -> call 911
    SELF_HARM = "self_harm"  # -> call or text 988 (US Suicide & Crisis Lifeline)


# The fixed replies. In a real product these are written with clinical and legal
# review and never changed casually. Tests check the key numbers are present.
SAFETY_REPLIES: dict[EmergencyCategory, str] = {
    EmergencyCategory.MEDICAL: (
        "This sounds like it could be a medical emergency. Please call 911 or go to "
        "the nearest emergency room right now. I'm a benefits assistant and can't give "
        "medical help. Emergency care is covered at any ER, in or out of network. "
        "Once you're safe, I'm here to help with claims or coverage."
    ),
    EmergencyCategory.SELF_HARM: (
        "I'm really sorry you're going through this. You don't have to face it alone. "
        "Please call or text 988 (Suicide & Crisis Lifeline) to talk to someone right "
        "now, any time. If you're in immediate danger, call 911. "
        "Your plan also covers mental health care, and I can help you find it whenever you're ready."
    ),
}


# PYTHON: frozen=True makes instances immutable (assigning a field raises an error)
# and hashable. A good default for small "result" objects that shouldn't change.
@dataclass(frozen=True)
class EmergencyResult:
    category: EmergencyCategory
    rule: str            # WHICH rule fired: essential for debugging and audits
    source: str = "rules"  # "rules" now; "classifier" arrives in step 2

    @property
    def reply(self) -> str:
        return SAFETY_REPLIES[self.category]


# =============================================================================
# The rules
# =============================================================================
# Each rule: (name, category, regex). The name is what gets logged and tested.
#
# PRESENT tense only for stroke / heart attack / seizure: "my dad HAD a stroke last
# year, is rehab covered?" is a benefits question, not an emergency happening now.
#
# PYTHON (regex refresher for the bits used here):
#   \b        word boundary: "\bod\b" matches "od" but not inside "odd"
#   (a|b)     either a or b
#   ?         the previous thing is optional: "breath(e|ing)?" , "can'?t" = can't / cant
#   \s+       one or more spaces;   (\w+\s+){0,3} = up to 3 words in between
#
# SECURITY/ROBUSTNESS: we normalise text first (lowercase, curly quotes -> straight,
# extra spaces collapsed) so "Can’t  BREATHE" still matches. Without that, a phone
# keyboard's curly apostrophe alone would silently defeat the rule.
_SUBJECT = r"(i|i'?m|he|he'?s|she|she'?s|they|they'?re|my \w+|someone)"

_RULES: list[tuple[str, EmergencyCategory, str]] = [
    # ---- Medical (911) ----
    ("chest_pain", EmergencyCategory.MEDICAL,
     r"\bchest\s+(pain|pressure|tightness|hurts?|is\s+tight)\b|\bpain\s+in\s+(my|his|her|the)\s+chest\b"),
    ("heart_attack", EmergencyCategory.MEDICAL,
     r"\b(having|think\s+\w+\s+(is\s+)?having)\s+a\s+heart\s+attack\b|\bheart\s+attack\s+(right\s+)?now\b"),
    ("cant_breathe", EmergencyCategory.MEDICAL,
     r"\b(can'?t|cannot|can\s+not|unable\s+to|struggling\s+to|trouble|difficulty)\s+breath(e|ing)?\b"
     r"|\bnot\s+breathing\b|\bshort(ness)?\s+of\s+breath\b|\bgasping\b"),
    ("stroke", EmergencyCategory.MEDICAL,
     r"\bhaving\s+a\s+stroke\b|\bsigns?\s+of\s+(a\s+)?stroke\b|\bslurred\s+speech\b|\bface\s+(is\s+)?droop"),
    ("unconscious", EmergencyCategory.MEDICAL,
     r"\bunconscious\b|\bunresponsive\b|\bpassed\s+out\b|\bwon'?t\s+wake\s+up\b|\bcollapsed\b"),
    ("severe_bleeding", EmergencyCategory.MEDICAL,
     r"\b(heavy|severe|uncontrolled)\s+bleeding\b|\bbleeding\s+(heavily|a\s+lot|badly)\b"
     r"|\bbleeding\s+(that\s+)?(won'?t|will\s+not|doesn'?t)\s+stop\b"),
    ("overdose", EmergencyCategory.MEDICAL,
     r"\boverdos(e|ed|ing)\b|\b(took|swallowed)\s+(a\s+(whole|full)\s+)?(bottle|handful)\b"),
    ("anaphylaxis", EmergencyCategory.MEDICAL,
     r"\bthroat\s+(is\s+)?(closing|swelling|swollen)\b|\banaphyla(xis|ctic)\b"),
    ("seizure", EmergencyCategory.MEDICAL,
     rf"\b{_SUBJECT}\s+(is\s+|are\s+)?having\s+a\s+seizure\b|\bseizing\b"),

    # ---- Self-harm (988) ----
    ("self_harm", EmergencyCategory.SELF_HARM,
     r"\b(kill|hurt|harm|cut)(ing)?\s+myself\b|\bself[\s-]?harm\b|\bsuicid(e|al)\b"
     r"|\bend\s+(my\s+life|it\s+all)\b|\bwant\s+to\s+die\b|\bdon'?t\s+want\s+to\s+(live|be\s+alive)\b"
     r"|\bno\s+reason\s+to\s+live\b|\bbetter\s+off\s+dead\b"),
]

# Compile once at import time, not on every message (compiling is the slow part).
# ORDER MATTERS: self-harm rules are checked first, because a message like
# "I overdosed on purpose, I want to die" needs the 988 reply (which also says 911).
_COMPILED: list[tuple[str, EmergencyCategory, re.Pattern[str]]] = sorted(
    [(name, cat, re.compile(pattern)) for name, cat, pattern in _RULES],
    key=lambda rule: rule[1] != EmergencyCategory.SELF_HARM,  # False sorts first
)


def _normalize(text: str) -> str:
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")  # curly -> straight
    # PYTHON: "\u2019" is an ESCAPE for the character ’ (right curly quote). Written as
    # an escape so it's visible in the code; the literal character looks identical to '.
    return re.sub(r"\s+", " ", text).strip()


def check_emergency(message: str) -> EmergencyResult | None:
    """Return the first matching emergency rule, or None if the message looks safe.

    KNOWN LIMITATION (accepted on purpose, recall over precision): no negation
    handling, so "I don't have chest pain, is cardiology covered?" is flagged.
    The member gets the safety reply and can rephrase. Handling negation with
    regex is fragile, and a missed emergency is far worse than this.
    """
    text = _normalize(message)
    for name, category, pattern in _COMPILED:
        # PYTHON: .search finds a match ANYWHERE in the text (.match only checks the start).
        if pattern.search(text):
            return EmergencyResult(category=category, rule=name)
    return None

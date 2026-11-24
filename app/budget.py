"""Tracks how much the LLM has cost today and enforces the daily spend cap.

CONCEPT (cost control): every LLM response reports its token USAGE (how many
input and output tokens it used). Cost = tokens x price per token. We add up
each call's cost and refuse new LLM calls once today's total reaches the budget.
That's the spend cap from the Architecture doc, enforced in OUR code rather
than relying only on the provider's dashboard.

LIMITATION (worth saying in interviews): this total lives in memory, so it
resets when the server restarts, and two server processes would each have
their own total. Production would keep it in a shared store such as Redis
(an atomic INCRBYFLOAT per day key) so every server sees the same number.
"""

from datetime import date

from app.config import get_settings


class SpendTracker:
    def __init__(self) -> None:
        self._day = date.today()
        self._spent_usd = 0.0

    def _roll_over_if_new_day(self) -> None:
        # A new day starts a fresh budget.
        if date.today() != self._day:
            self._day = date.today()
            self._spent_usd = 0.0

    # PYTHON: @property turns a METHOD into something that READS like an ATTRIBUTE.
    # Callers write   tracker.spent_today     (no parentheses)
    # instead of      tracker.spent_today()
    # but Python still runs this method's code every time it's read.
    #
    # Why not just a plain attribute, self.spent_today = 0.0? Because the value needs
    # a CHECK every time it's read: if it's a new day, reset to 0 first. A plain
    # attribute would happily return yesterday's total. A property runs that check
    # automatically, and callers never need to know it happens.
    #
    # Read-only by default: `tracker.spent_today = 5` raises AttributeError, because
    # we didn't define a setter. Only record() can change the total, so no other code
    # can accidentally bypass the spend cap by overwriting it.
    # (To allow assignment you'd add a second method decorated @spent_today.setter,
    # usually to validate the new value, e.g. reject negative amounts.)
    #
    # The leading underscore in self._spent_usd is a convention for "internal, don't
    # touch from outside". The property is the public, safe way to read it.
    #
    # Rule of thumb: use a property for something cheap that feels like DATA (a
    # value, a status). Use a normal method, with parentheses, for ACTIONS or slow
    # work, like record() here, or anything that calls a network or database.
    @property
    def spent_today(self) -> float:
        self._roll_over_if_new_day()
        return self._spent_usd

    def can_spend(self) -> bool:
        return self.spent_today < get_settings().daily_llm_budget_usd

    def record(self, input_tokens: int, output_tokens: int) -> float:
        """Add one LLM call's cost to today's total. Returns that call's cost in USD."""
        s = get_settings()
        cost = (input_tokens * s.llm_price_input_per_m
                + output_tokens * s.llm_price_output_per_m) / 1_000_000
        # PYTHON: 1_000_000 is just 1000000. Underscores in numbers are for readability.
        self._roll_over_if_new_day()
        self._spent_usd += cost
        return cost


# One shared tracker for the whole process (a module-level SINGLETON).
# CONCEPT (async safety): this += is safe without a lock because async code runs
# on ONE thread, and the event loop only switches tasks at an `await`. There is
# no `await` inside record(), so no other task can interleave halfway through it.
tracker = SpendTracker()

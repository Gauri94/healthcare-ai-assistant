"""App settings, loaded from environment variables and the .env file.

CONCEPT (12-factor apps): configuration that changes between environments
(your laptop, CI, production) lives in ENVIRONMENT VARIABLES, not in code.
Secrets especially: the API key stays in .env (git-ignored) locally, and in a
secrets manager in production. The code only ever says "give me the key".

CONCEPT (pydantic-settings): BaseSettings works like a Pydantic BaseModel, but
fills its fields from environment variables (matched by name, case-insensitive)
and from .env. Values are type-checked: DAILY_LLM_BUDGET_USD=abc fails at
startup with a clear error, not halfway through a member's chat.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SECURITY: SecretStr hides the value when printed or logged:
    #   print(settings.openrouter_api_key)  ->  **********
    # You must call .get_secret_value() to use it, which makes accidental leaks
    # (in logs, error pages, tracebacks) much less likely.
    # Default None = no key configured -> the app runs in rule-based fallback mode.
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # CONCEPT: the model is CONFIGURATION, not code. Switching to another model is
    # a one-line .env change, e.g. LLM_MODEL=openai/gpt-5-mini. This makes model
    # bake-offs (same evals, different models) trivial.
    llm_model: str = "anthropic/claude-haiku-5.5"

    # Prices in USD per 1 million tokens, from OpenRouter's model list (Nov 2026).
    # If you change llm_model, update these too, or the spend tracker will be wrong.
    llm_price_input_per_m: float = 0.10
    llm_price_output_per_m: float = 0.50

    # ---- Guardrails on cost and runaway behaviour ----
    # max_tokens caps the length of ONE reply (output tokens are the expensive ones).
    llm_max_tokens: int = 600
    # Max LLM calls in one agent loop. Stops a confused model from calling tools forever.
    max_agent_steps: int = 5
    # Give up on one LLM call after this many seconds.
    llm_timeout_s: float = 20.0
    # The app's own SPEND CAP: once today's LLM spend reaches this, the assistant
    # switches to rule-based fallback mode until tomorrow. A KILL SWITCH too:
    # set it to 0 to turn the LLM off instantly without a code change.
    daily_llm_budget_usd: float = 0.50


# PYTHON: @lru_cache (from functools) REMEMBERS a function's return value. Calling
# the function again with the SAME arguments returns the saved result instead of
# running the body again. The cache is keyed by the arguments:
#
#     @lru_cache
#     def slow_square(n):
#         print(f"computing {n}...")
#         return n * n
#
#     slow_square(4)   # prints "computing 4...", returns 16
#     slow_square(4)   # prints nothing: returns the remembered 16 instantly
#     slow_square(5)   # new argument, so it computes again
#
# LRU = "Least Recently Used": when the cache is full (128 entries by default),
# the entry used longest ago is thrown away first.
#
# CONCEPT: on a function with NO arguments there's only one possible key, so the
# body runs EXACTLY ONCE and every later call gets the same object back. That's a
# simple way to make a SINGLETON (one shared instance for the whole app). Here it
# means the .env file is read and validated on the first call only, not on every
# request.
#
# Two extra tools lru_cache adds to the function:
#   get_settings.cache_clear()  -> forget the cached value; the next call recomputes.
#                                  Tests use this to reload settings after changing
#                                  environment variables.
#   get_settings.cache_info()   -> hits, misses and current size of the cache.
#
# When NOT to use it:
#   - results that should change over time ("current time", "today's spend"): the
#     cache would keep returning a stale value. That's why SpendTracker doesn't use it.
#   - unhashable arguments (lists, dicts): arguments become dictionary keys, so they
#     must be hashable. Strings, numbers and tuples are fine.
#   - methods on objects, with care: the cache holds a reference to `self`, which can
#     keep objects in memory longer than you'd expect.
@lru_cache
def get_settings() -> Settings:
    return Settings()

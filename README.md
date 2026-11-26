# Healthcare AI Assistant

A member-facing health plan assistant that answers questions about claims, coverage and deductibles. It uses **synthetic data only**.

The core design rule is: **the LLM handles language, and plain code handles facts, identity and safety.** The LLM picks tools and words the reply. Authorization, arithmetic, emergency handling and PII redaction never depend on the LLM behaving.

## Run it

```bash
uv sync                          # install dependencies
uv run fastapi dev app/main.py   # start the API with auto-reload
```

Open http://127.0.0.1:8000/docs, click **Authorize** and paste `token-asha` or `token-ben`.

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer token-asha" -H "Content-Type: application/json" \
  -d '{"message": "Is a knee MRI (73721) covered, and how much deductible do I have left?"}'
```

## Project layout

| Path | What it does |
|---|---|
| `app/main.py` | FastAPI app: `/chat`, direct tool endpoints, `/health`, error handling |
| `app/safety.py` | Emergency detection before the LLM: fixed 911 / 988 replies |
| `app/auth.py` | Login token → `member_id` (the only source of identity) |
| `app/schemas.py` | Pydantic request/response models |
| `app/graph_agent.py` | LangGraph agent (default): graph of nodes, multi-turn memory via a checkpointer |
| `app/llm_agent.py` | Hand-written agent loop (`AGENT_ENGINE=loop`) and shared tool execution |
| `app/agent.py` | Rule-based fallback planner (no LLM) |
| `app/tool_specs.py` | Tool definitions shown to the LLM, and validation of its tool calls |
| `app/config.py`, `app/budget.py` | Settings from `.env`; daily spend cap |
| `app/tools.py` | Async tools with authorization inside each tool, plus `run_tool` timeouts |
| `app/data.py` | Fake member, claims and coverage data |
| `scripts/try_tools.py` | Async demos: `gather`, timeouts, partial failure, the blocking trap |
| `scripts/show_graph.py` | Prints the LangGraph agent as a Mermaid diagram |
| `tests/` | pytest suite with a scripted fake LLM (`conftest.py`) |

## Tests

```bash
uv run pytest -q
```

The suite has about 100 tests and runs in under a second, with no network and no LLM cost. A scripted **fake LLM** forces situations that are hard to trigger with a real model, and the tests check that the code still holds:

| Area | Examples |
|---|---|
| Security | A prompt-injected model requests another member's claim; the model smuggles `member_id` into tool arguments; one member reuses another's `conversation_id` |
| Robustness | Unknown tools, invalid JSON, missing arguments; one failing tool among several |
| Guardrails | Step limit on a model that never finishes; daily spend cap and kill switch; fallback on provider errors or a missing key |
| API | 401 without a valid token, 404 for other members' data (IDOR), 422 for invalid input |
| Concurrency | 10 simultaneous requests must overlap, which catches blocking calls in async code |
| Safety | Emergencies get the fixed reply and never reach the LLM; look-alike benefits questions are not flagged; known gaps are tracked with strict `xfail` |

Agent tests run against **both** engines, the LangGraph agent and the hand-written loop. Each protection was checked by deliberately breaking it and confirming that the matching tests fail.

## Roadmap

**Core:** what the portfolio and interview story need.

| # | Milestone | What it adds | Status |
|---|---|---|---|
| M1 | Async tools | Concurrent tool calls, timeouts, authorization inside tools | Done |
| M2 | FastAPI API | `/chat`, Pydantic validation, token auth dependency, IDOR-safe routes | Done |
| M3 | LLM agent | Tool calling via OpenRouter, parallel tool execution, retries, spend cap, no-LLM fallback; LangGraph version with multi-turn memory | Done |
| M5 | Test suite | Async tests, fake LLM, security, guardrail and concurrency tests | Done |
| M6 | Emergency detector | Rules + classifier, threshold tuned for a recall target, fixed safety reply | In progress: rules done |
| M8 | Policy RAG | Hybrid search (BM25 + embeddings), reranker, citations, abstention | |
| M10 | Evals and CI gate | Golden set (about 50 cases to start), LLM-judge rubric, GitHub Actions blocks regressions | |
| M11 | Deploy | Docker, hosted demo, rate limiting | |

**Later:** if time allows. The design for each is in the architecture doc.

| # | Milestone | What it adds |
|---|---|---|
| M4 | Streaming | Token-by-token replies over Server-Sent Events |
| M7 | Router | TF-IDF + logistic regression, macro-F1, calibration, clarify below a confidence threshold |
| M7+ | Router bake-off | Fine-tune DistilBERT in PyTorch and compare with M7 on macro-F1, latency and cost |
| M9 | PII redaction and audit log | Presidio redaction before LLM calls and logs, structured logs with request IDs |
| — | Conversation summaries | Summarize older turns instead of dropping them |

How the milestones map to interview topics:

- **AI / Gen AI engineer:** M3 agent and LangGraph, M8 RAG, M10 evals, cost controls
- **ML engineer:** M6 classifier, with metrics, calibration and threshold choice
- **SDET AI:** M5 test strategy, M10 golden set and CI gate, adversarial and prompt-injection tests

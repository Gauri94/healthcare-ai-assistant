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
| `app/auth.py` | Login token → `member_id` (the only source of identity) |
| `app/schemas.py` | Pydantic request/response models |
| `app/agent.py` | Plans tool calls, runs them concurrently, writes the reply |
| `app/tools.py` | Async tools with authorization inside each tool, plus `run_tool` timeouts |
| `app/data.py` | Fake member, claims and coverage data |
| `scripts/try_tools.py` | Async demos: `gather`, timeouts, partial failure, the blocking trap |

## MVP roadmap

| # | Milestone | What it adds | Status |
|---|---|---|---|
| M1 | Async tools | Concurrent tool calls, timeouts, authorization inside tools | Done |
| M2 | FastAPI API | `/chat`, Pydantic validation, token auth dependency, IDOR-safe routes | Done |
| M3 | LLM agent loop | Tool calling via OpenRouter, parallel tool execution, retries, spend cap, no-LLM fallback | Done |
| M4 | Streaming | Token-by-token replies over Server-Sent Events | Next |
| M5 | Test suite | pytest async tests, mocked LLM, security and concurrency tests | |
| M6 | Emergency detector | Rules + classifier, threshold tuned for a recall target, fixed safety reply | |
| M7 | Router | TF-IDF + logistic regression, macro-F1, calibration, clarify below a confidence threshold | |
| M7+ | Router bake-off (stretch) | Fine-tune DistilBERT in PyTorch and compare with M7 on macro-F1, latency and cost | |
| M8 | Policy RAG | Hybrid search (BM25 + embeddings), reranker, citations, abstention | |
| M9 | PII redaction and audit log | Presidio redaction before LLM calls and logs, structured logs with request IDs | |
| M10 | Evals and CI gate | Golden set (150–300 cases), LLM-judge rubric, GitHub Actions blocks regressions | |
| M11 | Deploy | Docker, hosted demo, rate limiting, spend cap, kill switch | |

The milestones map to interview topics like this:

- **AI / Gen AI engineer:** M3 agent loop, M4 streaming, M8 RAG, M10 evals, cost controls in M11
- **ML engineer:** M6 and M7 classifiers, with metrics, calibration and threshold choice, plus the PyTorch bake-off
- **SDET AI:** M5 test strategy, M10 golden set and CI gate, adversarial and prompt-injection tests, security tests for authorization

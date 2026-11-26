"""Concurrency test: catches the M1 BLOCKING TRAP at the API level.

CONCEPT: a blocking call (time.sleep, requests.get, a sync DB driver) inside async
code looks fine when you test ONE request. It only shows up when MANY requests
arrive at once: they queue behind each other instead of overlapping. So this test
fires many requests concurrently and checks the total time.

With real async code: 10 requests x 0.2s of tool latency all overlap  -> ~0.2s total.
If someone adds a blocking call:   they run one after another         -> ~2s total.
"""

import asyncio
import time

import httpx

from app import tools
from app.main import app
from tests.conftest import ASHA

TOOL_LATENCY_S = 0.2
N_REQUESTS = 10


async def test_concurrent_requests_overlap_instead_of_queueing(monkeypatch):
    # Turn simulated latency back ON for this test (conftest turns it off).
    async def real_latency(seconds: float) -> None:
        await asyncio.sleep(TOOL_LATENCY_S)
    monkeypatch.setattr(tools, "simulate_latency", real_latency)

    # CONCEPT: httpx.AsyncClient + ASGITransport sends requests straight into the
    # FastAPI app in-process (no network), AND lets us send them concurrently,
    # which the synchronous TestClient can't do.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.perf_counter()
        responses = await asyncio.gather(
            *(client.get("/me/deductible", headers=ASHA) for _ in range(N_REQUESTS))
        )
        elapsed = time.perf_counter() - start

    assert all(r.status_code == 200 for r in responses)
    # Serial would be N x 0.2 = 2.0s. Allow generous slack for slow CI machines,
    # but stay far below the serial time.
    assert elapsed < TOOL_LATENCY_S * N_REQUESTS / 3, f"requests queued instead of overlapping: {elapsed:.2f}s"

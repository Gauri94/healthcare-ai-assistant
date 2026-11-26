"""API tests: authentication, authorization (IDOR) and input validation over HTTP.

CONCEPT: TestClient calls the FastAPI app in-process, with no real server or
network, so these tests are fast. Using it in a `with` block also runs the app's
lifespan (startup and shutdown), like a real server would.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import ASHA, BEN, text_reply


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---- Authentication: who are you? ----

@pytest.mark.parametrize("headers", [
    {},                                          # no header at all
    {"Authorization": "Bearer wrong-token"},     # unknown token
    {"Authorization": "token-asha"},             # missing "Bearer " scheme
])
def test_requests_without_a_valid_token_get_401(client, headers):
    assert client.get("/me/deductible", headers=headers).status_code == 401
    assert client.post("/chat", headers=headers, json={"message": "hi"}).status_code == 401


def test_each_member_sees_only_their_own_data(client):
    assert client.get("/me/deductible", headers=ASHA).json()["remaining"] == 850.0
    assert client.get("/me/deductible", headers=BEN).json()["remaining"] == 3100.0


# ---- Authorization: IDOR ----

def test_member_cannot_fetch_another_members_claim(client):
    response = client.get("/me/claims/C-2001", headers=ASHA)  # Ben's claim
    assert response.status_code == 404
    assert "Office visit" not in response.text  # none of Ben's data leaks


def test_owner_can_fetch_their_claim(client):
    response = client.get("/me/claims/C-2001", headers=BEN)
    assert response.status_code == 200
    assert response.json()["service"] == "Office visit"


# ---- Input validation ----

@pytest.mark.parametrize("path", ["/me/claims/abc", "/me/claims/C-12", "/coverage/7372", "/coverage/abcde"])
def test_malformed_ids_are_rejected_before_any_tool_runs(client, path):
    assert client.get(path, headers=ASHA).status_code == 422


@pytest.mark.parametrize("body", [
    {"message": ""},                                   # empty
    {"message": "x" * 2001},                           # over max_length
    {"message": "my deductible", "member_id": "M456"},  # smuggled identity field
    {"message": "hi", "conversation_id": "../../etc"},  # unsafe characters
    {},                                                # missing message
])
def test_invalid_chat_bodies_get_422(client, body):
    assert client.post("/chat", headers=ASHA, json=body).status_code == 422


# ---- Plumbing ----

def test_health_needs_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_chat_response_shape(client, fake_llm):
    fake_llm.script = [text_reply("Hello! How can I help?")]
    data = client.post("/chat", headers=ASHA, json={"message": "hi"}).json()
    assert data["reply"] == "Hello! How can I help?"
    assert data["mode"] == "llm"
    assert data["tool_calls"] == []
    assert data["cost_usd"] > 0
    assert data["request_id"]

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app import pipeline
from app.main import MAX_HISTORY, MAX_QUESTION_CHARS, app

client = TestClient(app)  # not a context manager: the startup data load is skipped


def test_chat_returns_a_cited_answer(fakes):
    resp = client.post("/chat", json={"question": "hybrid Apple revenue and why"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["behavior"] == "answer" and body["route"] == "hybrid"
    assert body["citations"][0]["url"].endswith("doc.htm")
    assert body["tool"]["rows"][0]["value"] == "391035000000"
    assert body["unsupported"] == []  # "$391 billion" is in the cited chunk
    assert body["run_id"] and body["ms"]["total"] >= 0


def test_chat_passes_history(fakes):
    history = [{"role": "user", "content": "Apple revenue 2023?"},
               {"role": "assistant", "content": "$383 billion [21]."}]  # fmt: skip
    resp = client.post("/chat", json={"question": "And 2022?", "history": history})
    assert resp.status_code == 200
    assert resp.json()["standalone_question"] == "metric Apple revenue fiscal 2022"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": "   "},
        {"question": "x" * (MAX_QUESTION_CHARS + 1)},
        {"question": "ok?", "history": [{"role": "system", "content": "hi"}]},
        {"question": "ok?", "history": [{"role": "user", "content": ""}]},
        {"question": "ok?",
         "history": [{"role": "user", "content": "hi"}] * (MAX_HISTORY + 1)},
    ],
)  # fmt: skip
def test_bad_requests_get_400(fakes, body):
    resp = client.post("/chat", json=body)
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad request" and resp.json()["details"]


def test_model_failure_gets_502(fakes, monkeypatch):
    def fail(question, model):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://x"))

    monkeypatch.setattr(pipeline, "route", fail)
    resp = client.post("/chat", json={"question": "metric Apple revenue"})
    assert resp.status_code == 502
    assert resp.json() == {"error": "model call failed", "detail": "APIConnectionError"}


def test_cors_allows_the_local_frontend():
    resp = client.options(
        "/chat",
        headers={"Origin": "http://localhost:3000",
                 "Access-Control-Request-Method": "POST"},
    )  # fmt: skip
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_health():
    assert client.get("/health").json() == {"status": "ok"}

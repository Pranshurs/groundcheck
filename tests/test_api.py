"""API tests via FastAPI's TestClient — proves the service boots and answers in fallback mode."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_reports_backend():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] in {"model", "heuristic"}


def test_check_returns_scored_result():
    r = client.post("/api/check", json={
        "source": "France's capital and largest city is Paris.",
        "answer": "Paris is the capital of France.",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"grounded", "hallucinated"}
    assert 0.0 <= body["grounded_score"] <= 1.0
    assert "hallucination_score" in body


def test_check_batch():
    r = client.post("/api/check_batch", json={"items": [
        {"source": "The cat sat on the mat.", "answer": "A cat was on the mat."},
        {"source": "It rained on Tuesday.", "answer": "It was sunny all week."},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 2


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "GroundCheck" in r.text

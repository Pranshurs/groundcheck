"""API tests via FastAPI's TestClient, on the explicit heuristic backend (no weights needed)."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from groundcheck import api

README = Path(__file__).resolve().parent.parent / "README.md"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_BACKEND", "heuristic")
    with TestClient(api.app) as c:
        yield c


def test_readme_uvicorn_target_is_importable():
    # Guards against the README drifting from the real module again.
    targets = re.findall(r"uvicorn\s+([\w.]+):(\w+)", README.read_text(encoding="utf-8"))
    assert targets, "README no longer documents a uvicorn command"
    for module, attr in targets:
        assert hasattr(importlib.import_module(module), attr), f"{module}:{attr}"


def test_readme_documents_real_routes():
    routes = {r.path for r in api.app.routes}
    for path in re.findall(r"POST (/[\w/]+)", README.read_text(encoding="utf-8")):
        assert path in routes, f"README documents {path}, app serves {sorted(routes)}"


def test_health_reports_backend(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "heuristic"
    assert body["model"]["backend"] == "heuristic"


def test_check_returns_scored_result(client):
    r = client.post("/api/check", json={
        "source": "France's capital and largest city is Paris.",
        "answer": "Paris is the capital of France.",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"grounded", "hallucinated"}
    assert 0.0 <= body["grounded_score"] <= 1.0
    assert body["backend"] == "heuristic"


def test_check_batch(client):
    r = client.post("/api/check_batch", json={"items": [
        {"source": "The cat sat on the mat.", "answer": "A cat was on the mat."},
        {"source": "It rained on Tuesday.", "answer": "It was sunny all week."},
    ]})
    assert r.status_code == 200
    assert len(r.json()) == 2


@pytest.mark.parametrize("payload", [
    {"source": "", "answer": "x"},
    {"source": "x", "answer": ""},
    {"source": "x"},
    {"source": "x" * (api.MAX_SOURCE_CHARS + 1), "answer": "x"},
    {"source": "x", "answer": "x" * (api.MAX_ANSWER_CHARS + 1)},
])
def test_check_rejects_invalid_input(client, payload):
    assert client.post("/api/check", json=payload).status_code == 422


def test_batch_limits(client):
    one = {"source": "a", "answer": "a"}
    assert client.post("/api/check_batch", json={"items": []}).status_code == 422
    too_many = {"items": [one] * (api.MAX_BATCH_ITEMS + 1)}
    assert client.post("/api/check_batch", json=too_many).status_code == 422


def test_index_page_served_from_package(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "GroundCheck" in r.text

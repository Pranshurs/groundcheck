"""Model tier: loads the pinned published weights and checks real inference.

    pytest -m model          # downloads ~600 MB on first run (HF cache)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.model

transformers = pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

from groundcheck import GroundCheck, InputTooLongError, ModelUnavailableError, api  # noqa: E402
from groundcheck.config import DEFAULT_MAX_LENGTH, DEFAULT_MODEL_REVISION  # noqa: E402
from training.data import MANUAL_CASES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = json.loads((ROOT / "MODEL_PROVENANCE.json").read_text())["published_metrics"]["manual5"]


@pytest.fixture(scope="module")
def gc() -> GroundCheck:
    return GroundCheck(backend="model", device="cpu")


def test_loads_the_pinned_revision(gc):
    info = gc.model_info()
    assert gc.backend == "model"
    assert info["resolved_revision"] == DEFAULT_MODEL_REVISION
    assert info["architecture"] == "ModernBertForSequenceClassification"
    assert info["max_length"] == DEFAULT_MAX_LENGTH


@pytest.mark.parametrize("i", range(len(MANUAL_CASES)))
def test_manual_cases_match_the_published_run(gc, i):
    # Labels and P(hallucinated) as logged by the training run (metrics.json on the Hub).
    case, published = MANUAL_CASES[i], PUBLISHED[i]
    r = gc.check(case["source"], case["answer"])
    assert r.backend == "model"
    assert r.label == case["label"] == published["got"]
    assert r.hallucination_score == pytest.approx(published["p_hall"], abs=0.02)


def test_direction_on_a_minimal_edit(gc):
    src = "In Q3, revenue increased 12% year-over-year to $2.1 billion."
    assert gc.check(src, "Revenue rose 12% in Q3.").grounded
    assert not gc.check(src, "Revenue fell 12% in Q3.").grounded


def test_batch_matches_single(gc):
    items = [{"source": c["source"], "answer": c["answer"]} for c in MANUAL_CASES]
    batch = gc.check_many(items, batch_size=3)
    for item, b in zip(items, batch):
        assert b.grounded_score == pytest.approx(gc.check(item["source"], item["answer"]).grounded_score, abs=1e-4)


def test_long_source_is_truncated_not_the_answer(gc):
    fact = "The bridge opened to traffic in 1932."
    filler = " ".join(["The weather in the region is usually mild in spring."] * 200)
    r = gc.check(fact + " " + filler, "The bridge opened in 1932.")
    assert r.grounded  # supporting fact is at the start and survives source truncation


def test_answer_longer_than_budget_is_rejected(gc):
    with pytest.raises(InputTooLongError, match="Split it"):
        gc.check("short source", "word " * 2500)


def test_api_returns_422_for_overlong_answer(monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_BACKEND", "model")
    with TestClient(api.app) as client:
        assert client.get("/api/health").json()["model"]["resolved_revision"] == DEFAULT_MODEL_REVISION
        r = client.post("/api/check", json={"source": "short", "answer": "word " * 2500})
        assert r.status_code == 422 and "tokens" in r.json()["detail"]


def test_model_with_swapped_labels_is_refused(gc, tmp_path):
    # Copy the real model, flip id2label, and make sure GroundCheck will not load it.
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download("Pranshurs/groundcheck-modernbert", revision=DEFAULT_MODEL_REVISION))
    for f in src.iterdir():
        target = tmp_path / f.name
        if f.name == "model.safetensors":
            target.symlink_to(f.resolve())
        else:
            shutil.copy(f, target)
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["id2label"] = {"0": "hallucinated", "1": "grounded"}
    cfg["label2id"] = {"hallucinated": 0, "grounded": 1}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ModelUnavailableError, match="expects"):
        GroundCheck(model_path=str(tmp_path), revision=None)

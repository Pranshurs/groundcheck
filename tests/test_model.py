"""Tests for the detector's heuristic fallback (the path that runs without torch)."""

from __future__ import annotations

from groundcheck import GroundCheck
from groundcheck.config import DetectorSettings


def _heuristic() -> GroundCheck:
    return GroundCheck(DetectorSettings(model_path="", backend="heuristic",
                                        threshold=0.5, max_length=512, device="cpu"))


def test_backend_is_heuristic_without_a_model():
    gc = _heuristic()
    assert gc.backend == "heuristic"


def test_grounded_answer_scores_higher_than_hallucinated():
    gc = _heuristic()
    source = "France is a country in Western Europe. Its capital and largest city is Paris."
    grounded = gc.check(source, "Paris is the capital of France.")
    hallucinated = gc.check(source, "Berlin is the capital of France and home to the Eiffel Tower.")
    assert grounded.grounded_score > hallucinated.grounded_score


def test_score_in_unit_interval_and_label_consistent():
    gc = _heuristic()
    r = gc.check("The sky is blue.", "The sky is blue today.")
    assert 0.0 <= r.grounded_score <= 1.0
    assert r.grounded == (r.label == "grounded")
    assert abs(r.hallucination_score - (1 - r.grounded_score)) < 1e-6


def test_empty_answer_is_trivially_grounded():
    gc = _heuristic()
    assert gc.check("anything at all", "").grounded_score == 1.0


def test_check_many_matches_single_calls():
    gc = _heuristic()
    items = [
        {"source": "The cat sat on the mat.", "answer": "A cat was on the mat."},
        {"source": "Water boils at 100C.", "answer": "Water boils at 50C in space hotels."},
    ]
    batch = gc.check_many(items)
    assert len(batch) == 2
    single = gc.check(items[0]["source"], items[0]["answer"])
    assert batch[0].grounded_score == single.grounded_score


def test_result_to_dict_roundtrip():
    gc = _heuristic()
    d = gc.check("a b c", "a b c").to_dict()
    assert set(["label", "grounded", "grounded_score", "backend",
                "latency_ms", "hallucination_score"]).issubset(d.keys())

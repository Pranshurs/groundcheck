"""Tests for the mock LLM judge (offline path used by the benchmark)."""

from __future__ import annotations

from groundcheck.config import JudgeSettings
from groundcheck.judge import MockJudge, _parse_judgement, build_judge


def _mock_settings() -> JudgeSettings:
    return JudgeSettings(mode="mock", base_url="", api_key="", model="mock",
                         temperature=0.0, price_in_per_mtok=0.59, price_out_per_mtok=0.79)


def test_build_judge_returns_mock_without_key():
    assert isinstance(build_judge(_mock_settings()), MockJudge)


def test_mock_judge_reports_token_usage_for_cost_model():
    j = MockJudge()
    r = j.check("The capital of France is Paris.", "Paris is the capital of France.")
    assert r.prompt_tokens > 0
    assert r.completion_tokens > 0
    assert 0.0 <= r.grounded_score <= 1.0


def test_mock_judge_is_deterministic():
    j = MockJudge()
    a = j.check("a b c d e", "a b c")
    b = j.check("a b c d e", "a b c")
    assert a.grounded == b.grounded and a.grounded_score == b.grounded_score


def test_parse_judgement_reads_json():
    grounded, score = _parse_judgement('{"grounded": true, "confidence": 0.9}')
    assert grounded is True
    assert score == 0.9


def test_parse_judgement_flips_confidence_for_not_grounded():
    grounded, score = _parse_judgement('{"grounded": false, "confidence": 0.8}')
    assert grounded is False
    # expressed as P(grounded) => 1 - 0.8
    assert abs(score - 0.2) < 1e-6


def test_parse_judgement_tolerates_prose_wrapping():
    grounded, score = _parse_judgement('Sure! Here is my answer: {"grounded": true, "confidence": 1.0} done')
    assert grounded is True

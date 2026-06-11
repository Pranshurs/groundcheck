"""Tests for the scoring metrics — these guard the numbers the project lives or dies on."""

from __future__ import annotations

from eval import metrics


def test_perfect_predictions():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    assert metrics.accuracy(y_true, y_pred) == 1.0
    assert metrics.f1_score(y_true, y_pred) == 1.0
    assert metrics.balanced_accuracy(y_true, y_pred) == 1.0


def test_confusion_counts():
    # positive class == 1 (hallucinated)
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]  # tp=1, fn=1, fp=1, tn=1
    assert metrics.confusion(y_true, y_pred) == (1, 1, 1, 1)


def test_precision_recall_f1_known_value():
    y_true = [1, 1, 1, 0]
    y_pred = [1, 1, 0, 0]  # tp=2, fp=0, fn=1 -> P=1.0, R=2/3, F1=0.8
    p, r, f1 = metrics.precision_recall_f1(y_true, y_pred)
    assert p == 1.0
    assert round(r, 3) == 0.667
    assert round(f1, 3) == 0.8


def test_wilson_interval_brackets_point_estimate():
    lo, hi = metrics.wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0


def test_accuracy_ci_widens_with_small_n():
    lo_small, hi_small = metrics.accuracy_ci([1, 0], [1, 0])
    lo_big, hi_big = metrics.accuracy_ci([1, 0] * 100, [1, 0] * 100)
    # both perfect accuracy, but the small sample has a wider (lower) lower bound
    assert lo_small < lo_big


def test_bootstrap_f1_ci_is_ordered_and_deterministic():
    y_true = [1, 0, 1, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
    lo1, hi1 = metrics.bootstrap_f1_ci(y_true, y_pred, n_boot=500, seed=0)
    lo2, hi2 = metrics.bootstrap_f1_ci(y_true, y_pred, n_boot=500, seed=0)
    assert lo1 <= hi1
    assert (lo1, hi1) == (lo2, hi2)  # same seed => reproducible


def test_judge_cost_per_1k():
    # 1000 tokens in, 10 out, at $1/Mtok in and $2/Mtok out:
    # per check = 1000/1e6*1 + 10/1e6*2 = 0.00102 ; x1000 = 1.02
    cost = metrics.judge_cost_per_1k(1000, 10, 1.0, 2.0)
    assert round(cost, 2) == 1.02


def test_latency_summary():
    s = metrics.latency_summary([10, 20, 30, 40, 100])
    assert s.p50_ms == 30
    assert s.mean_ms == 40.0


def test_full_report_keys():
    rep = metrics.full_report([1, 0, 1, 0], [1, 0, 0, 0])
    for key in ("accuracy", "accuracy_ci95", "f1", "f1_ci95", "balanced_accuracy", "confusion"):
        assert key in rep
    assert rep["positive_class"] == "hallucinated"

"""End-to-end test of the benchmark harness on the offline sample set (heuristic + mock)."""

from __future__ import annotations

from eval import benchmark

DATA = "eval/cases/sample_grounding.jsonl"


def test_sample_dataset_loads_with_both_labels():
    rows = benchmark.load_dataset(DATA)
    assert len(rows) >= 10
    y = benchmark._y_true(rows)
    assert 0 in y and 1 in y  # both grounded and hallucinated present


def test_full_harness_produces_two_rows():
    rows = benchmark.load_dataset(DATA)
    y_true = benchmark._y_true(rows)
    det = benchmark.run_detector(rows, threshold=0.5)
    jud = benchmark.run_judge(rows)
    table = benchmark.build_rows(y_true, det, jud, vps_usd_per_hr=0.011)

    assert len(table) == 2
    for row in table:
        for key in ("system", "accuracy", "f1", "f1_ci95", "latency", "cost_per_1k_usd"):
            assert key in row
    md = benchmark.to_markdown(table)
    assert md.startswith("| System |")


def test_detector_beats_random_on_sample():
    # The heuristic should clear 0.5 accuracy on the (easy-ish) sample set.
    rows = benchmark.load_dataset(DATA)
    y_true = benchmark._y_true(rows)
    det = benchmark.run_detector(rows, threshold=0.5)
    from eval import metrics
    assert metrics.accuracy(y_true, det["y_pred"]) >= 0.5

"""Metrics for the benchmark — with the statistical rigour that's the point of this repo.

Convention: the **positive class is "hallucinated"** (label 1), because the task is
*detecting* hallucinations, matching how RAGTruth / LettuceDetect report detection F1.
So precision/recall/F1 describe how well a detector catches unsupported answers.

Two kinds of confidence interval, used deliberately:

* **Wilson** for accuracy — accuracy is a proportion of independent correct/incorrect
  calls, so a Wilson score interval is exactly right and needs no resampling.
* **Bootstrap (percentile)** for F1 — F1 is a ratio of sums, not a binomial proportion,
  so a Wilson interval would be wrong. We resample examples with replacement instead.

A point estimate over a few thousand examples without an interval is misleading; every
headline number in this project ships with one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def confusion(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) with positive class == 1 (hallucinated)."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    tp, fp, fn, _ = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def f1_score(y_true: list[int], y_pred: list[int]) -> float:
    return precision_recall_f1(y_true, y_pred)[2]


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Mean of per-class recall — robust to the class imbalance in grounding data."""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0  # recall on hallucinated
    tnr = tn / (tn + fp) if (tn + fp) else 0.0  # recall on grounded
    return (tpr + tnr) / 2


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def accuracy_ci(y_true: list[int], y_pred: list[int], z: float = 1.96) -> tuple[float, float]:
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return wilson_interval(correct, len(y_true), z)


def bootstrap_f1_ci(
    y_true: list[int],
    y_pred: list[int],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for F1 (positive class == hallucinated)."""
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    scores: list[float] = []
    idx = range(n)
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in idx]
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        scores.append(f1_score(yt, yp))
    scores.sort()
    lo = scores[int((alpha / 2) * n_boot)]
    hi = scores[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


@dataclass
class LatencySummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float


def latency_summary(latencies_ms: list[float]) -> LatencySummary:
    if not latencies_ms:
        return LatencySummary(0.0, 0.0, 0.0)
    s = sorted(latencies_ms)
    mean = sum(s) / len(s)
    p50 = s[int(0.50 * (len(s) - 1))]
    p95 = s[int(0.95 * (len(s) - 1))]
    return LatencySummary(round(mean, 2), round(p50, 2), round(p95, 2))


def judge_cost_per_1k(
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    price_in_per_mtok: float,
    price_out_per_mtok: float,
) -> float:
    """$ to run 1,000 judge checks at the given average token usage and prices."""
    per_check = (
        avg_prompt_tokens / 1_000_000 * price_in_per_mtok
        + avg_completion_tokens / 1_000_000 * price_out_per_mtok
    )
    return round(per_check * 1000, 4)


def full_report(y_true: list[int], y_pred: list[int]) -> dict:
    """Accuracy/precision/recall/F1/balanced-accuracy with CIs — one call for the table."""
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    acc_lo, acc_hi = accuracy_ci(y_true, y_pred)
    f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred)
    tp, fp, fn, tn = confusion(y_true, y_pred)
    return {
        "n": len(y_true),
        "accuracy": round(acc, 4),
        "accuracy_ci95": [round(acc_lo, 4), round(acc_hi, 4)],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f1_ci95": [round(f1_lo, 4), round(f1_hi, 4)],
        "balanced_accuracy": round(balanced_accuracy(y_true, y_pred), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "positive_class": "hallucinated",
    }

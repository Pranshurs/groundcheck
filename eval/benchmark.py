"""Head-to-head benchmark: GroundCheck vs an LLM-as-a-judge.

This is the deliverable that turns "I trained a model" into "I proved it's worth using".
On the same held-out set it measures, for both the detector and the judge:

    accuracy · precision · recall · F1   (with 95% CIs)
    latency  (p50 / p95)
    $ per 1,000 checks

and prints a Markdown table you can paste straight into the README, plus a JSON file
under eval/results/.

Run offline (mock judge, heuristic detector — proves the harness works with zero setup):

    python -m eval.benchmark --data eval/cases/sample_grounding.jsonl

Real numbers (set a trained model + a live judge key in .env):

    GROUNDCHECK_MODEL_PATH=./artifacts/groundcheck-modernbert \\
    LLM_API_KEY=...  python -m eval.benchmark --data data/ragtruth_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from groundcheck.config import get_detector_settings, get_judge_settings
from groundcheck.judge import build_judge
from groundcheck.model import GroundCheck

from . import metrics


def load_dataset(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _y_true(rows: list[dict]) -> list[int]:
    # positive class == hallucinated
    return [1 if str(r["label"]).lower().startswith("halluc") else 0 for r in rows]


def run_detector(rows: list[dict], threshold: float) -> dict:
    settings = get_detector_settings()
    settings.threshold = threshold
    gc = GroundCheck(settings)
    items = [{"source": r["source"], "answer": r["answer"], "question": r.get("question")} for r in rows]
    t0 = time.perf_counter()
    results = gc.check_many(items)
    wall_s = time.perf_counter() - t0
    y_pred = [0 if r.grounded else 1 for r in results]  # predicted hallucinated == 1
    latencies = [r.latency_ms for r in results]
    throughput = len(rows) / wall_s if wall_s else 0.0
    return {
        "name": f"GroundCheck ({gc.backend})",
        "backend": gc.backend,
        "y_pred": y_pred,
        "latencies_ms": latencies,
        "throughput_per_s": round(throughput, 2),
    }


def run_judge(rows: list[dict]) -> dict:
    js = get_judge_settings()
    judge = build_judge(js)
    y_pred, latencies, p_tokens, c_tokens = [], [], [], []
    for r in rows:
        jr = judge.check(r["source"], r["answer"], r.get("question"))
        y_pred.append(0 if jr.grounded else 1)
        latencies.append(jr.latency_ms)
        p_tokens.append(jr.prompt_tokens)
        c_tokens.append(jr.completion_tokens)
    avg_p = sum(p_tokens) / len(p_tokens) if p_tokens else 0.0
    avg_c = sum(c_tokens) / len(c_tokens) if c_tokens else 0.0
    cost = metrics.judge_cost_per_1k(avg_p, avg_c, js.price_in_per_mtok, js.price_out_per_mtok)
    return {
        "name": f"LLM judge ({js.model}, {js.mode})",
        "mode": js.mode,
        "y_pred": y_pred,
        "latencies_ms": latencies,
        "cost_per_1k_usd": cost,
        "avg_prompt_tokens": round(avg_p, 1),
        "avg_completion_tokens": round(avg_c, 1),
    }


def _detector_cost_per_1k(throughput_per_s: float, vps_usd_per_hr: float) -> float:
    """Self-hosted CPU cost: time to run 1,000 checks × the box's hourly price."""
    if throughput_per_s <= 0:
        return 0.0
    seconds_per_1k = 1000.0 / throughput_per_s
    return round(seconds_per_1k / 3600.0 * vps_usd_per_hr, 4)


def build_rows(y_true: list[int], detector: dict, judge: dict | None, vps_usd_per_hr: float) -> list[dict]:
    table: list[dict] = []

    det_report = metrics.full_report(y_true, detector["y_pred"])
    det_lat = metrics.latency_summary(detector["latencies_ms"])
    table.append(
        {
            "system": detector["name"],
            **det_report,
            "latency": det_lat.__dict__,
            "cost_per_1k_usd": _detector_cost_per_1k(detector["throughput_per_s"], vps_usd_per_hr),
        }
    )

    if judge is not None:
        j_report = metrics.full_report(y_true, judge["y_pred"])
        j_lat = metrics.latency_summary(judge["latencies_ms"])
        table.append(
            {
                "system": judge["name"],
                **j_report,
                "latency": j_lat.__dict__,
                "cost_per_1k_usd": judge["cost_per_1k_usd"],
            }
        )
    return table


def to_markdown(table: list[dict]) -> str:
    head = (
        "| System | Acc | Acc 95% CI | F1 | F1 95% CI | Bal.Acc | Latency p50 (ms) | $/1k |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for row in table:
        acc_ci = f"[{row['accuracy_ci95'][0]:.3f}, {row['accuracy_ci95'][1]:.3f}]"
        f1_ci = f"[{row['f1_ci95'][0]:.3f}, {row['f1_ci95'][1]:.3f}]"
        lines.append(
            f"| {row['system']} | {row['accuracy']:.3f} | {acc_ci} | {row['f1']:.3f} | "
            f"{f1_ci} | {row['balanced_accuracy']:.3f} | {row['latency']['p50_ms']:.1f} | "
            f"${row['cost_per_1k_usd']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GroundCheck against an LLM judge.")
    parser.add_argument("--data", default="eval/cases/sample_grounding.jsonl", help="JSONL test set.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of examples.")
    parser.add_argument("--threshold", type=float, default=0.5, help="grounded_score decision threshold.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge baseline.")
    parser.add_argument("--vps-usd-per-hr", type=float, default=0.011, help="Self-host price (≈$8/mo).")
    parser.add_argument("--out", default="eval/results", help="Directory for the JSON result.")
    args = parser.parse_args()

    rows = load_dataset(args.data, args.limit)
    y_true = _y_true(rows)
    print(f"Loaded {len(rows)} examples from {args.data} "
          f"({sum(y_true)} hallucinated / {len(y_true) - sum(y_true)} grounded)\n")

    detector = run_detector(rows, args.threshold)
    judge = None if args.no_judge else run_judge(rows)
    table = build_rows(y_true, detector, judge, args.vps_usd_per_hr)

    md = to_markdown(table)
    print(md + "\n")
    if judge and judge.get("mode") == "mock":
        print("NOTE: judge ran in MOCK mode — set LLM_API_KEY for real judge numbers.\n")
    if detector["backend"] == "heuristic":
        print("NOTE: detector ran on the HEURISTIC fallback — set GROUNDCHECK_MODEL_PATH "
              "to benchmark the trained model.\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "dataset": args.data,
        "n": len(rows),
        "threshold": args.threshold,
        "results": table,
        "markdown": md,
    }
    out_path = out_dir / f"benchmark-{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

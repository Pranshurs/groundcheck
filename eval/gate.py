"""Regression gate: does a reproduction run still behave like the reference model?

    python -m eval.reproduce --subset eval/baselines/gate_subset.json --out /tmp/gate.json
    python -m eval.gate /tmp/gate.json                       # exit 1 on any failed check

    python -m eval.gate --make-baseline eval/reports/<full-run>.json   # maintainers only

The reference is a committed full reproduction run (eval/baselines/v2-reference.json):
its per-row predictions on a fixed, stratified subset plus its subset metrics.

Checks, and why each threshold is what it is:

1. Provenance. The run must use the pinned revision, the recorded weight hash and the
   manifest's data hashes. Without this, no comparison is meaningful.
2. Per-row agreement >= 98% of subset predictions, mean |Δ score| <= 0.01. This is the
   sensitive check. Re-running on different hardware (Kaggle P100 vs Apple M1 CPU) moved
   1 of 2,500 RAGTruth predictions (0.04%) and the manual-case scores by < 0.02, so 98%
   leaves ~50x headroom for numeric noise while any change to tokenisation, truncation,
   label mapping or weights moves far more rows.
3. Metric floors per suite: F1 and accuracy may drop at most 0.03 below the reference on
   the same rows. On a 200-row slice the bootstrap CI of F1 is wide (about ±0.06), so this
   catches broad regressions, not subtle ones; check 2 covers those.
4. Per-class recall may drop at most 0.05 for either class. A model collapsing toward one
   label can keep accuracy plausible on an imbalanced slice; per-class recall can't hide it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "eval" / "baselines" / "v2-reference.json"
SUBSET = ROOT / "eval" / "baselines" / "gate_subset.json"
PROVENANCE = ROOT / "MODEL_PROVENANCE.json"
MANIFEST = ROOT / "data" / "manifest.v2.json"
GATE_SUITES = {"ragtruth": 200, "vitaminc": 100}
THRESHOLDS = {"min_agreement": 0.98, "max_mean_abs_score_delta": 0.01,
              "max_metric_drop": 0.03, "max_class_recall_drop": 0.05}


def _scores(result_path: Path) -> dict:
    return json.loads(result_path.with_suffix(".scores.json").read_text())


def make_subset(seed: int = 0) -> dict:
    """Stratified (by label) fixed sample of row indices per suite."""
    rng = random.Random(seed)
    files = {"ragtruth": "ragtruth_test.jsonl", "vitaminc": "vitaminc_test.jsonl"}
    out = {}
    for suite, n in GATE_SUITES.items():
        rows = [json.loads(line) for line in (ROOT / "data" / files[suite]).open(encoding="utf-8")]
        by_label: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by_label.setdefault(r["label"], []).append(i)
        picked = []
        for _label, idx in sorted(by_label.items()):
            k = round(n * len(idx) / len(rows))
            picked += rng.sample(idx, k)
        out[suite] = sorted(picked)
    return out


def make_baseline(full_result: Path) -> None:
    result = json.loads(full_result.read_text())
    scores = _scores(full_result)
    subset = json.loads(SUBSET.read_text())
    from .reproduce import suite_metrics, load_suite

    manifest = json.loads(MANIFEST.read_text())
    ref = {"source_run": full_result.name, "model": result["model"], "data": result["data"],
           "environment": result["environment"], "thresholds": THRESHOLDS, "suites": {}}
    for suite, idx in subset.items():
        full_idx = scores[suite]["index"]
        assert full_idx == list(range(len(full_idx))), "baseline must come from a full run"
        sub_scores = [scores[suite]["grounded_score"][i] for i in idx]
        rows = load_suite(suite, ROOT / "data", manifest, allow_unverified=False)
        ref["suites"][suite] = {
            "index": idx,
            "grounded_score": sub_scores,
            "metrics": suite_metrics([rows[i] for i in idx], sub_scores, result["protocol"]["threshold"]),
        }
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(ref, indent=1) + "\n")
    print(f"wrote {BASELINE.relative_to(ROOT)}")


def check(result_path: Path, baseline_path: Path = BASELINE) -> list[tuple[str, bool, str]]:
    result = json.loads(result_path.read_text())
    scores = _scores(result_path)
    ref = json.loads(baseline_path.read_text())
    prov = json.loads(PROVENANCE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    threshold = result["protocol"]["threshold"]
    t = ref["thresholds"]
    out: list[tuple[str, bool, str]] = []

    m = result["model"]
    out.append(("provenance: revision", m.get("resolved_revision") == prov["revision"],
                f"{m.get('resolved_revision')} vs {prov['revision']}"))
    out.append(("provenance: weights sha256",
                m.get("weights_sha256") == prov["files"]["model.safetensors"]["sha256"],
                str(m.get("weights_sha256"))[:16]))
    data_ok = all(result["data"]["files"].get(f) == e["sha256"] for f, e in manifest["files"].items()
                  if f in result["data"]["files"])
    out.append(("provenance: data hashes", data_ok, result["data"]["manifest_sha256"][:16]))
    out.append(("protocol: max_length", result["protocol"]["max_length"] == ref["model"]["max_length"],
                f"{result['protocol']['max_length']} vs {ref['model']['max_length']}"))

    for suite, r in ref["suites"].items():
        got = scores.get(suite)
        if got is None or got["index"] != r["index"]:
            out.append((f"{suite}: same rows as reference", False, "suite missing or rows differ"))
            continue
        pairs = list(zip(got["grounded_score"], r["grounded_score"]))
        agree = sum((a >= threshold) == (b >= threshold) for a, b in pairs) / len(pairs)
        delta = sum(abs(a - b) for a, b in pairs) / len(pairs)
        out.append((f"{suite}: prediction agreement", agree >= t["min_agreement"],
                    f"{agree:.4f} (min {t['min_agreement']})"))
        out.append((f"{suite}: mean |Δ score|", delta <= t["max_mean_abs_score_delta"],
                    f"{delta:.5f} (max {t['max_mean_abs_score_delta']})"))
        now, before = result["suites"][suite], r["metrics"]
        for key in ("f1", "accuracy"):
            out.append((f"{suite}: {key}", now[key] >= before[key] - t["max_metric_drop"],
                        f"{now[key]:.4f} vs ref {before[key]:.4f}"))
        for key in ("recall_hallucinated", "recall_grounded"):
            out.append((f"{suite}: {key}", now[key] >= before[key] - t["max_class_recall_drop"],
                        f"{now[key]:.4f} vs ref {before[key]:.4f}"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GroundCheck regression gate")
    ap.add_argument("result", nargs="?", help="result JSON from eval.reproduce --subset ...")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--make-subset", action="store_true")
    ap.add_argument("--make-baseline", metavar="FULL_RESULT")
    args = ap.parse_args(argv)

    if args.make_subset:
        SUBSET.parent.mkdir(parents=True, exist_ok=True)
        SUBSET.write_text(json.dumps(make_subset()) + "\n")
        print(f"wrote {SUBSET.relative_to(ROOT)}")
        return 0
    if args.make_baseline:
        make_baseline(Path(args.make_baseline))
        return 0
    if not args.result:
        ap.error("give a result JSON")

    checks = check(Path(args.result), Path(args.baseline))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:38} {detail}")
    failed = [c for c in checks if not c[1]]
    print(f"\ngate: {'PASSED' if not failed else f'FAILED ({len(failed)} check(s))'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

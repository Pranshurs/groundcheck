"""Re-derive GroundCheck's published numbers from pinned weights and pinned public data.

    python -m training.build_data --verify          # build data/ and check it against the manifest
    python -m eval.reproduce                        # full run: RAGTruth + VitaminC + flipped + manual
    python -m eval.reproduce --subset eval/baselines/gate_subset.json --out result.json   # CI slice

Every result file records the model revision and weight hash, the data hashes, the code
commit and the environment, so a number can always be traced back to what produced it.
Inputs are checked against data/manifest.v2.json before anything runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from groundcheck.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from groundcheck.model import GroundCheck

from . import metrics

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.v2.json"
SUITES = {
    "ragtruth": "ragtruth_test.jsonl",
    "vitaminc": "vitaminc_test.jsonl",
    "flipped_hallucinated": "flipped_test_hallucinated.jsonl",
    "flipped_original": "flipped_test_original.jsonl",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_suite(name: str, data_dir: Path, manifest: dict, allow_unverified: bool) -> list[dict]:
    fname = SUITES[name]
    path = data_dir / fname
    want = manifest["files"][fname]["sha256"]
    got = _sha256(path)
    if got != want and not allow_unverified:
        raise SystemExit(f"{fname}: sha256 {got[:16]} does not match manifest {want[:16]}; "
                         f"rebuild with `python -m training.build_data`")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_suite(gc: GroundCheck, rows: list[dict], batch_size: int) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    results = gc.check_many(rows, batch_size=batch_size)
    return [r.grounded_score for r in results], time.perf_counter() - t0


def suite_metrics(rows: list[dict], scores: list[float], threshold: float) -> dict:
    y_true = [1 if r["label"] == "hallucinated" else 0 for r in rows]
    y_pred = [0 if s >= threshold else 1 for s in scores]
    p, r, f1 = metrics.precision_recall_f1(y_true, y_pred)
    tp, fp, fn, tn = metrics.confusion(y_true, y_pred)
    out = {
        "n": len(rows),
        "accuracy": round(metrics.accuracy(y_true, y_pred), 4),
        "precision": round(p, 4),
        "recall_hallucinated": round(r, 4),
        "recall_grounded": round(tn / (tn + fp), 4) if tn + fp else None,
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    if 0 < sum(y_true) < len(y_true):
        lo, hi = metrics.bootstrap_f1_ci(y_true, y_pred, n_boot=2000, seed=0)
        out["f1_ci95"] = [round(lo, 4), round(hi, 4)]
    return out


def _git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _environment() -> dict:
    import torch
    import transformers

    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    elif Path("/proc/cpuinfo").exists():
        cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                    if ln.startswith("model name")), cpu)
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "torch_threads": torch.get_num_threads()}


def _loaded_weights_sha256(gc: GroundCheck) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = Path(gc.settings.model_path) / "model.safetensors"
    if not path.exists():
        cached = try_to_load_from_cache(gc.settings.model_path, "model.safetensors",
                                        revision=gc.settings.revision)
        path = Path(cached) if isinstance(cached, str) else None
    return _sha256(path) if path else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-path", default=DEFAULT_MODEL_ID)
    ap.add_argument("--revision", default=None, help="default: pinned revision for the published model")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--suites", nargs="+", default=list(SUITES) + ["manual"],
                    choices=list(SUITES) + ["manual"])
    ap.add_argument("--subset", default=None, help="JSON {suite: [row indices]} to score only those rows")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=None, help="result JSON path (default: eval/reports/<timestamp>.json)")
    ap.add_argument("--allow-unverified-data", action="store_true")
    args = ap.parse_args(argv)

    revision = args.revision or (DEFAULT_MODEL_REVISION if args.model_path == DEFAULT_MODEL_ID else None)
    data_dir = Path(args.data)
    manifest = json.loads((data_dir / MANIFEST.name).read_text())
    subset = json.loads(Path(args.subset).read_text()) if args.subset else None

    t_load = time.perf_counter()
    gc = GroundCheck(model_path=args.model_path, revision=revision, max_length=args.max_length,
                     threshold=args.threshold, backend="model", device="cpu")
    load_s = time.perf_counter() - t_load

    result: dict = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code": _git_state(),
        "model": {**gc.model_info(), "weights_sha256": _loaded_weights_sha256(gc)},
        "data": {"manifest_sha256": _sha256(data_dir / MANIFEST.name),
                 "files": {SUITES[s]: manifest["files"][SUITES[s]]["sha256"] for s in SUITES}},
        "protocol": {"threshold": args.threshold, "max_length": args.max_length,
                     "positive_class": "hallucinated", "bootstrap": "2000 resamples, seed 0, percentile 95%",
                     "subset": args.subset},
        "environment": _environment(),
        "timing": {"model_load_seconds": round(load_s, 2)},
        "suites": {},
    }
    scores_out: dict = {}

    for name in [s for s in args.suites if s != "manual"]:
        rows = load_suite(name, data_dir, manifest, args.allow_unverified_data)
        index = list(range(len(rows)))
        if subset is not None:
            if name not in subset:
                continue
            index = subset[name]
            rows = [rows[i] for i in index]
        scores, seconds = score_suite(gc, rows, args.batch_size)
        result["suites"][name] = suite_metrics(rows, scores, args.threshold)
        result["timing"][f"{name}_seconds"] = round(seconds, 1)
        scores_out[name] = {"index": index, "grounded_score": [round(s, 5) for s in scores]}
        print(f"{name:22} {json.dumps(result['suites'][name])}", flush=True)

    if "manual" in args.suites and subset is None:
        from training.data import MANUAL_CASES

        manual = []
        for case in MANUAL_CASES:
            r = gc.check(case["source"], case["answer"])
            manual.append({"answer": case["answer"][:60], "expected": case["label"], "got": r.label,
                           "p_hallucinated": round(r.hallucination_score, 3), "pass": r.label == case["label"]})
        result["suites"]["manual"] = {"passed": sum(m["pass"] for m in manual), "n": len(manual), "cases": manual}
        print(f"{'manual':22} {result['suites']['manual']['passed']}/{len(manual)}", flush=True)

    out = Path(args.out) if args.out else ROOT / "eval" / "reports" / (
        f"reproduce-{result['created_utc'][:19].replace(':', '')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    out.with_suffix(".scores.json").write_text(json.dumps(scores_out) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

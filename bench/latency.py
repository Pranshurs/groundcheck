"""Measure GroundCheck CPU latency on this machine, on real benchmark inputs.

    python -m bench.latency                       # writes bench/results/<machine>-<date>.json
    python -m bench.latency --runs 100 --threads 4

Model load time and inference latency are reported separately. Inputs are real rows from
the pinned test sets, grouped by the token length of the (question + source, answer) pair:

    short   <= 128 tokens     (VitaminC-style claim/evidence pairs)
    medium  129-512 tokens
    long    513-2048 tokens   (typical RAGTruth documents)
    xlong   > 2048 tokens     (source truncated at the default max_length)

Run it at the package default (2048) and at the published protocol (--max-length 512).

Results describe *this* hardware only; they are not a general CPU claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# sysctl/subprocess forks after tokenisation; keep the tokenizers pool from warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
BUCKETS = (("short", 0, 128), ("medium", 129, 512), ("long", 513, 2048), ("xlong", 2049, 10**9))


def _cpu() -> dict:
    info = {"machine": platform.machine(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        def q(key: str) -> str:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()

        info.update(model=q("machdep.cpu.brand_string"), memory_gb=round(int(q("hw.memsize")) / 2**30))
    elif Path("/proc/cpuinfo").exists():
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        info["model"] = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("model name")), None)
    return info


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=60, help="timed single-request runs per bucket")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8, help="for the throughput measurement")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's choice)")
    ap.add_argument("--max-length", type=int, default=None, help="default: the package default (2048)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    t_import = time.perf_counter()
    import torch
    import transformers
    from groundcheck import GroundCheck
    import_s = time.perf_counter() - t_import
    if args.threads:
        torch.set_num_threads(args.threads)

    t_load = time.perf_counter()
    overrides = {"max_length": args.max_length} if args.max_length else {}
    gc = GroundCheck(backend="model", device="cpu", **overrides)
    load_s = time.perf_counter() - t_load

    rows = []
    for f in ("ragtruth_test.jsonl", "vitaminc_test.jsonl"):
        rows += [json.loads(line) for line in (ROOT / "data" / f).open(encoding="utf-8")]
    tok = gc._tokenizer
    def pair_len(r):
        prem = f"{r['question']}\n\n{r['source']}" if r["question"] else r["source"]
        return len(tok(prem, r["answer"])["input_ids"])

    rng = random.Random(0)
    rng.shuffle(rows)
    pools: dict[str, list[dict]] = {name: [] for name, _, _ in BUCKETS}
    for r in rows:
        n = pair_len(r)
        for name, lo, hi in BUCKETS:
            if lo <= n <= hi and len(pools[name]) < args.runs + args.warmup:
                pools[name].append(r)
        if all(len(p) >= args.runs + args.warmup for p in pools.values()):
            break

    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hardware": _cpu(),
        "software": {"python": platform.python_version(), "os": platform.platform(),
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "torch_threads": torch.get_num_threads()},
        "model": gc.model_info(),
        "system_load_avg_1_5_15m_at_start": [round(x, 2) for x in os.getloadavg()],
        "load": {"import_seconds": round(import_s, 2), "model_load_seconds": round(load_s, 2),
                 "note": "load time includes reading weights from the local HF cache; not a download"},
        "single_request_ms": {},
        "batch_throughput": {},
    }

    for name, lo, hi in BUCKETS:
        pool = pools[name]
        for r in pool[: args.warmup]:
            gc.check(r["source"], r["answer"], r["question"] or None)
        timed = pool[args.warmup: args.warmup + args.runs]
        lat = []
        for r in timed:
            t0 = time.perf_counter()
            gc.check(r["source"], r["answer"], r["question"] or None)
            lat.append((time.perf_counter() - t0) * 1000)
        lens = [min(pair_len(r), gc.settings.max_length) for r in timed]
        report["single_request_ms"][name] = {
            "tokens_range": [lo, hi if hi < 10**9 else None], "n": len(lat),
            "model_tokens_median": statistics.median(lens),
            "p50": round(_pct(lat, 0.50), 1), "p95": round(_pct(lat, 0.95), 1),
            "mean": round(statistics.fmean(lat), 1), "max": round(max(lat), 1),
        }
        t0 = time.perf_counter()
        gc.check_many([{"source": r["source"], "answer": r["answer"], "question": r["question"]} for r in timed],
                      batch_size=args.batch_size)
        wall = time.perf_counter() - t0
        report["batch_throughput"][name] = {"batch_size": args.batch_size, "rows": len(timed),
                                            "rows_per_second": round(len(timed) / wall, 2)}
        print(f"{name:7} p50 {report['single_request_ms'][name]['p50']:7.1f} ms   "
              f"p95 {report['single_request_ms'][name]['p95']:7.1f} ms   "
              f"batch{args.batch_size} {report['batch_throughput'][name]['rows_per_second']:6.2f} rows/s", flush=True)

    hw = report["hardware"]
    slug = (hw.get("model") or hw["machine"]).lower().replace(" ", "-").replace("(r)", "").replace("(tm)", "")
    out = Path(args.out) if args.out else ROOT / "bench" / "results" / f"{slug}-maxlen{gc.settings.max_length}-{report['created_utc'][:10]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"model load {load_s:.2f} s (import {import_s:.2f} s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

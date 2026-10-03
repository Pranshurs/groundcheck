"""``groundcheck`` console command — check whether an answer is grounded in a source.

Installed as a script via pyproject (``groundcheck ...``); also used by the repo's run.py.

    groundcheck --source "Paris is the capital of France." --answer "France's capital is Paris."
    groundcheck --source-file ctx.txt --answer-file ans.txt --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import BACKENDS
from .model import GroundCheck, ModelUnavailableError


def _read(inline: str | None, path: str | None, what: str) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    if inline is not None:
        return inline
    raise SystemExit(f"Provide --{what} or --{what}-file")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check whether an answer is grounded in a source.")
    p.add_argument("--source", default=None)
    p.add_argument("--answer", default=None)
    p.add_argument("--question", default=None)
    p.add_argument("--source-file", default=None)
    p.add_argument("--answer-file", default=None)
    p.add_argument("--json", action="store_true", help="Print the full JSON result.")
    p.add_argument("--backend", choices=BACKENDS, default=None,
                   help="model (default) or heuristic; overrides GROUNDCHECK_BACKEND")
    p.add_argument("--model", default=None, help="HF id or local dir (default: published v2)")
    p.add_argument("--revision", default=None, help="HF revision to load")
    args = p.parse_args(argv)

    source = _read(args.source, args.source_file, "source")
    answer = _read(args.answer, args.answer_file, "answer")

    overrides = {k: v for k, v in (("backend", args.backend), ("model_path", args.model),
                                   ("revision", args.revision)) if v is not None}
    try:
        detector = GroundCheck(**overrides)
    except (ModelUnavailableError, ValueError) as exc:
        print(f"groundcheck: {exc}", file=sys.stderr)
        return 2
    result = detector.check(source, answer, args.question)

    if args.json:
        print(json.dumps({**result.to_dict(), "model": detector.model_info()}, indent=2))
        return 0

    mark = "✓ GROUNDED" if result.grounded else "✗ HALLUCINATED"
    print(f"\n{mark}   (backend: {result.backend})")
    print(f"grounded score      : {result.grounded_score:.3f}")
    print(f"hallucination risk  : {result.hallucination_score:.3f}")
    print(f"latency             : {result.latency_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

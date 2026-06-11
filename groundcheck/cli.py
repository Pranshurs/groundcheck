"""``groundcheck`` console command — check whether an answer is grounded in a source.

Installed as a script via pyproject (``groundcheck ...``); also used by the repo's run.py.

    groundcheck --source "Paris is the capital of France." --answer "France's capital is Paris."
    groundcheck --source-file ctx.txt --answer-file ans.txt --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import GroundCheck


def _read(inline: str | None, path: str | None, what: str) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    if inline is not None:
        return inline
    raise SystemExit(f"Provide --{what} or --{what}-file")


def main() -> None:
    p = argparse.ArgumentParser(description="Check whether an answer is grounded in a source.")
    p.add_argument("--source", default=None)
    p.add_argument("--answer", default=None)
    p.add_argument("--question", default=None)
    p.add_argument("--source-file", default=None)
    p.add_argument("--answer-file", default=None)
    p.add_argument("--json", action="store_true", help="Print the full JSON result.")
    args = p.parse_args()

    source = _read(args.source, args.source_file, "source")
    answer = _read(args.answer, args.answer_file, "answer")

    result = GroundCheck().check(source, answer, args.question)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    mark = "✓ GROUNDED" if result.grounded else "✗ HALLUCINATED"
    print(f"\n{mark}   (backend: {result.backend})")
    print(f"grounded score      : {result.grounded_score:.3f}")
    print(f"hallucination risk  : {result.hallucination_score:.3f}")
    print(f"latency             : {result.latency_ms:.1f} ms")


if __name__ == "__main__":
    main()

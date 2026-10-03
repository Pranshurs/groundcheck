"""Build (or verify) the v2 train/val/test JSONL files from pinned public datasets.

    python -m training.build_data                 # build into data/ and write data/manifest.v2.json
    python -m training.build_data --verify        # rebuild and compare against the committed manifest

Nothing here is downloaded from private storage: RAGTruth and VitaminC come from the
Hugging Face Hub at the revisions pinned in training/configs/v2.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .data import DataConfig, build_splits, summarize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "training" / "configs" / "v2.json"
DEFAULT_OUT = ROOT / "data"
MANIFEST_NAME = "manifest.v2.json"


def canonical_jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for r in rows
    )


def load_sources(cfg: dict):
    from datasets import load_dataset  # heavy import kept local

    ds = cfg["datasets"]
    ragtruth = load_dataset(ds["ragtruth"]["repo_id"], revision=ds["ragtruth"]["revision"])
    vitaminc = load_dataset(ds["vitaminc"]["repo_id"], revision=ds["vitaminc"]["revision"])
    return ragtruth, vitaminc


def build(cfg: dict) -> tuple[dict[str, bytes], dict]:
    ragtruth, vitaminc = load_sources(cfg)
    splits = build_splits(DataConfig.from_dict(cfg["data"]), ragtruth, vitaminc)
    files, entries = {}, {}
    for name, rows in splits.items():
        blob = canonical_jsonl(rows)
        files[f"{name}.jsonl"] = blob
        entries[f"{name}.jsonl"] = {"sha256": hashlib.sha256(blob).hexdigest(), **summarize(rows)}
    manifest = {
        "config": cfg["name"],
        "datasets": cfg["datasets"],
        "data_config": cfg["data"],
        "source_rows": {
            "ragtruth": {k: len(v) for k, v in ragtruth.items()},
            "vitaminc": {k: len(v) for k, v in vitaminc.items()},
        },
        "files": entries,
    }
    return files, manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and fail if any file differs from the committed manifest")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    files, manifest = build(cfg)

    if args.verify:
        expected = json.loads((out / MANIFEST_NAME).read_text())
        bad = []
        for name, entry in manifest["files"].items():
            want = expected["files"].get(name, {}).get("sha256")
            status = "ok" if want == entry["sha256"] else "MISMATCH"
            if status != "ok":
                bad.append(name)
            print(f"{status:8} {name:34} n={entry['n']:6}  {entry['sha256'][:16]}")
        missing = set(expected["files"]) - set(manifest["files"])
        bad += sorted(missing)
        if bad:
            print(f"FAILED: {len(bad)} file(s) differ: {', '.join(bad)}", file=sys.stderr)
            return 1
        print("all files match the manifest")

    out.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (out / name).write_bytes(blob)
    if not args.verify:
        (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name, entry in manifest["files"].items():
        print(f"{name:34} {json.dumps({k: v for k, v in entry.items() if k != 'sha256'})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

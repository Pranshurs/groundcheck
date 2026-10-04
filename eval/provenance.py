"""Pin and verify exactly which published weights this repo evaluates.

    python -m eval.provenance --verify   # download at the pinned revision, check every hash
    python -m eval.provenance --write    # regenerate MODEL_PROVENANCE.json (maintainers)

The record ties the Hub revision to SHA-256 hashes of every file that affects predictions,
plus the training arguments saved by the run itself (training_args.bin), so a reviewer can
check that the weights, the recipe in training/configs/v2.json and the published numbers
belong together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from groundcheck.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "MODEL_PROVENANCE.json"
FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "training_args.bin", "metrics.json")
TRAINING_ARGS = ("num_train_epochs", "per_device_train_batch_size", "learning_rate",
                 "weight_decay", "warmup_ratio", "lr_scheduler_type", "seed", "fp16",
                 "gradient_checkpointing", "group_by_length", "optim", "max_grad_norm")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(repo_id: str, revision: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    paths = {f: hf_hub_download(repo_id, f, revision=revision) for f in FILES}
    hub = {s.rfilename: s for s in HfApi().model_info(repo_id, revision=revision, files_metadata=True).siblings}
    files = {}
    for name, path in paths.items():
        digest = sha256(path)
        lfs = getattr(hub[name], "lfs", None)
        if lfs is not None and lfs.sha256 != digest:
            raise SystemExit(f"{name}: local sha256 {digest} != Hub LFS sha256 {lfs.sha256}")
        files[name] = {"sha256": digest, "bytes": Path(path).stat().st_size}

    config = json.loads(Path(paths["config.json"]).read_text())
    record = {
        "repo_id": repo_id,
        "revision": revision,
        "weights_commit": "0c7dd0636c0d58b5e3865db8f1deee5a5acfeb6c",
        "note": "998cec35 changed only README.md; every file below is identical at 0c7dd063.",
        "architecture": config["architectures"],
        "id2label": config["id2label"],
        "base_model": config.get("_name_or_path"),
        "trained_with_transformers": config.get("transformers_version"),
        "files": files,
    }
    try:  # training_args.bin is a pickle written by the original run; read a few fields
        import torch

        args = torch.load(paths["training_args.bin"], weights_only=False)
        record["training_args"] = {k: str(getattr(args, k)) for k in TRAINING_ARGS}
    except ImportError:
        pass
    record["published_metrics"] = json.loads(Path(paths["metrics.json"]).read_text())["results"]
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    if args.write:
        record = collect(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION)
        RECORD.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {RECORD.name}")
        return 0

    expected = json.loads(RECORD.read_text())
    if expected["revision"] != DEFAULT_MODEL_REVISION:
        print(f"package pins {DEFAULT_MODEL_REVISION}, record has {expected['revision']}", file=sys.stderr)
        return 1
    actual = collect(expected["repo_id"], expected["revision"])
    bad = [f for f in FILES if actual["files"][f]["sha256"] != expected["files"][f]["sha256"]]
    for f in FILES:
        print(f"{'ok' if f not in bad else 'MISMATCH':8} {f:24} {actual['files'][f]['sha256'][:16]}")
    for key in ("architecture", "id2label", "training_args"):
        if key in actual and actual[key] != expected.get(key):
            bad.append(key)
            print(f"MISMATCH {key}: {actual[key]} != {expected.get(key)}")
    if bad:
        return 1
    print("provenance verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

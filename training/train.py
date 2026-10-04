"""Fine-tune ModernBERT-base into the GroundCheck v2 classifier.

    python -m training.build_data                                  # writes data/*.jsonl
    python -m training.train --out artifacts/groundcheck-v2        # GPU strongly recommended
    python -m eval.reproduce --model-path artifacts/groundcheck-v2 # score it

This is the recipe that produced the published weights (Kaggle P100, ~54 min for
3 epochs), cleaned up: settings come from training/configs/v2.json instead of constants,
the base model is pinned to a Hub revision, and the Kaggle-specific pip pinning moved to
requirements-train.txt. ``--max-train-rows``/``--epochs`` exist for smoke runs only; a run
using them is not the v2 recipe and its metadata says so.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL2ID = {"grounded": 0, "hallucinated": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def premise(question: str, source: str) -> str:
    # Must match groundcheck.model._encode: question (if any), blank line, source.
    return f"{question}\n\n{source}" if question else source


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the GroundCheck v2 classifier.")
    ap.add_argument("--config", default=str(ROOT / "training" / "configs" / "v2.json"))
    ap.add_argument("--data", default=str(ROOT / "data"), help="directory written by build_data")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "groundcheck-v2"))
    ap.add_argument("--max-train-rows", type=int, default=None, help="smoke runs only")
    ap.add_argument("--epochs", type=float, default=None, help="smoke runs only")
    ap.add_argument("--max-length", type=int, default=None, help="smoke runs only")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import transformers
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    cfg = json.loads(Path(args.config).read_text())
    tcfg = dict(cfg["train"])
    smoke = any(v is not None for v in (args.max_train_rows, args.epochs, args.max_length))
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.max_length is not None:
        tcfg["max_length"] = args.max_length
    set_seed(tcfg["seed"])

    data_dir = Path(args.data)
    train_rows = read_jsonl(data_dir / "train.jsonl")[: args.max_train_rows]
    val_rows = read_jsonl(data_dir / "val.jsonl")[: (args.max_train_rows and max(16, args.max_train_rows // 4))]

    base = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(base["repo_id"], revision=base["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        base["repo_id"], revision=base["revision"], num_labels=2, id2label=ID2LABEL,
        label2id=LABEL2ID, attn_implementation=tcfg["attn_implementation"], reference_compile=False,
    )

    def encode(batch):
        enc = tok([premise(q, s) for q, s in zip(batch["question"], batch["source"])],
                  batch["answer"], truncation="only_first", max_length=tcfg["max_length"])
        enc["labels"] = [LABEL2ID[label] for label in batch["label"]]
        return enc

    def to_dataset(rows):
        cols = ["question", "source", "answer", "label"]
        return Dataset.from_list([{k: r[k] for k in cols} for r in rows]).map(
            encode, batched=True, remove_columns=cols)

    def metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, -1)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"accuracy": float((pred == labels).mean()), "precision": p, "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0}

    cuda = torch.cuda.is_available()
    out = Path(args.out)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["train_batch_size"],
        per_device_eval_batch_size=tcfg["eval_batch_size"], learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"], warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler"], eval_strategy="epoch", save_strategy="no",
        logging_steps=50, report_to="none", gradient_checkpointing=tcfg["gradient_checkpointing"],
        fp16=cuda and tcfg["fp16_on_cuda"], group_by_length=tcfg["group_by_length"],
        seed=tcfg["seed"], use_cpu=not cuda,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=to_dataset(train_rows),
                      eval_dataset=to_dataset(val_rows), processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    started = time.time()
    trainer.train()
    val = trainer.evaluate()

    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    manifest = json.loads((data_dir / "manifest.v2.json").read_text())
    meta = {
        "recipe": cfg["name"],
        "smoke_run": smoke,
        "train_config": tcfg,
        "base_model": base,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "data_sha256": {k: v["sha256"] for k, v in manifest["files"].items() if k in ("train.jsonl", "val.jsonl")},
        "val_metrics": val,
        "seconds": round(time.time() - started, 1),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cuda": cuda,
                        "device": torch.cuda.get_device_name(0) if cuda else platform.processor()},
    }
    (out / "training_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"val": val, "smoke_run": smoke}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Public data pipeline for GroundCheck v2.

Cleaned from the Kaggle script that trained the published weights
(``Pranshurs/groundcheck-modernbert``, HF revision 0c7dd063 / card 998cec35). The sampling
order, sizes, label mapping and hard-negative rules are unchanged, with one deliberate
deviation, documented in ``training/README.md``:

* The original iterated over Python ``set`` objects when choosing which fact to flip.
  String-set order depends on ``PYTHONHASHSEED``, which is randomised per process, so the
  original hard negatives cannot be regenerated bit-for-bit. Here every such set is sorted,
  which makes the build deterministic but means the regenerated hard negatives (and anything
  sampled after them from the same RNG) are not guaranteed identical to the Kaggle run.

Everything that does not touch that RNG path - the RAGTruth train/val/test rows and the
VitaminC test rows - is reproduced exactly.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

LABELS = ("grounded", "hallucinated")

# ----------------------------------------------------------------------------- RAGTruth
_QUESTION_KEYS = ("question", "query", "prompt", "instruction")
_SOURCE_KEYS = ("source", "context", "reference", "passages", "source_info", "prompt_context", "documents")
_ANSWER_KEYS = ("answer", "response", "output", "generation", "model_output", "model_response")


def _pick(row: dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, "", [], {}):
            v = row[k]
            return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
    return None


def _is_hallucinated(row: dict) -> Optional[bool]:
    p = row.get("hallucination_labels_processed")
    if isinstance(p, dict):
        return int(p.get("evident_conflict", 0) or 0) > 0 or int(p.get("baseless_info", 0) or 0) > 0
    hl = row.get("hallucination_labels")
    if isinstance(hl, str):
        try:
            return len(json.loads(hl)) > 0
        except ValueError:
            return hl.strip() not in ("", "[]", "null", "none")
    if isinstance(hl, (list, tuple)):
        return len(hl) > 0
    return None


def ragtruth_example(row: dict) -> Optional[dict]:
    """One RAGTruth response -> one example; hallucinated if any span is annotated."""
    source, answer = _pick(row, _SOURCE_KEYS), _pick(row, _ANSWER_KEYS)
    if not source or not answer:
        return None
    hall = _is_hallucinated(row)
    if hall is None:
        return None
    return {
        "question": _pick(row, _QUESTION_KEYS) or "",
        "source": source,
        "answer": answer,
        "label": "hallucinated" if hall else "grounded",
    }


# ----------------------------------------------------------------------------- VitaminC
def vitaminc_example(row: dict) -> Optional[dict]:
    """SUPPORTS -> grounded; REFUTES and NOT ENOUGH INFO -> hallucinated."""
    if not row.get("claim") or not row.get("evidence"):
        return None
    return {
        "question": "",
        "source": row["evidence"],
        "answer": row["claim"],
        "label": "grounded" if row["label"] == "SUPPORTS" else "hallucinated",
    }


# ----------------------------------------------------------------------------- hard negatives
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DIRECTIONS = [("increased", "decreased"), ("increase", "decrease"), ("increasing", "decreasing"),
              ("rose", "fell"), ("rise", "fall"), ("rising", "falling"), ("higher", "lower"),
              ("more than", "less than"), ("gained", "lost"), ("growth", "decline"),
              ("grew", "shrank"), ("above", "below"), ("surged", "plunged"), ("expanded", "contracted")]
_ENTITY_STOP = {"The", "A", "An", "In", "On", "At", "It", "He", "She", "They", "We", "You", "I",
                "This", "That", "These", "Those", "However", "According", "Meanwhile", "After",
                "Before", "When", "While", "If", "As", "For", "But", "And", "Or", "So", "Also",
                "There", "Here", "What", "Who", "Why", "How", "Its", "His", "Her", "Their", "Our",
                "Yes", "No", "Not", "New", "One", "Two", "First", "Second", "Mr", "Mrs", "Ms", "Dr",
                "Next", "Click", "Then", "Step", "Note", "Page", "Home", "Sure", "Welcome", "Summary",
                "Once", "Now", "Finally", "Overall", "Open", "Select", "Choose", "Enter", "Press"}
_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})*\b")


def _sub_token(text: str, old: str, new: str, numeric: bool = False) -> Optional[str]:
    # Whole-token replacement. Numeric mode also refuses digits/commas/periods at the edges
    # so "12" never matches inside "120" or "1,200".
    if numeric:
        pat = re.compile(r"(?<![\w.,$%])" + re.escape(old) + r"(?![\w.,%])")
    else:
        pat = re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)")
    return pat.sub(new, text) if pat.search(text) else None


def _format_like(orig: str, val: float) -> str:
    pct, dollar = orig.endswith("%"), orig.startswith("$")
    core = orig.rstrip("%").lstrip("$")
    dec = len(core.split(".")[1]) if "." in core else 0
    s = f"{val:,.{dec}f}" if "," in core else f"{val:.{dec}f}"
    return ("$" if dollar else "") + s + ("%" if pct else "")


def flip_number(ans: str, ctx: str, rng: random.Random):
    cands = sorted(t for t in set(_NUM_RE.findall(ans))
                   if t in ctx and not _YEAR_RE.fullmatch(t.strip("$%"))
                   and float(t.strip("$%").replace(",", "")) > 0)
    rng.shuffle(cands)
    for tok in cands:
        v = float(tok.strip("$%").replace(",", ""))
        opts = []
        if tok.endswith("%") and v == int(v) and 10 <= v <= 98 and int(str(int(v))[::-1]) != int(v):
            opts.append(float(str(int(v))[::-1]))  # 94% -> 49%
        opts += [round(v * f, 6) for f in (0.5, 1.5, 2, 3) if round(v * f, 6) != v]
        opts += [v + d for d in (max(1, round(v * 0.25)), -max(1, round(v * 0.25))) if v + d > 0]
        new = _format_like(tok, rng.choice(opts))
        if new != tok:
            out = _sub_token(ans, tok, new, numeric=True)
            if out:
                return out, f"number:{tok}->{new}"
    return None


def flip_direction(ans: str, rng: random.Random):
    pairs = [(a, b) for x, y in DIRECTIONS for a, b in ((x, y), (y, x))]
    rng.shuffle(pairs)
    for a, b in pairs:
        for src, dst in ((a, b), (a.capitalize(), b.capitalize())):
            out = _sub_token(ans, src, dst)
            if out:
                return out, f"direction:{src}->{dst}"
    return None


def flip_date(ans: str, ctx: str, rng: random.Random):
    years = [y.group(0) for y in _YEAR_RE.finditer(ans) if y.group(0) in ctx]
    months = [m for m in MONTHS if re.search(r"\b" + m + r"\b", ans) and m in ctx]
    cands = [("y", y) for y in sorted(set(years))] + [("m", m) for m in months]
    rng.shuffle(cands)
    for kind, tok in cands:
        if kind == "y":
            new = str(int(tok) + rng.choice([-3, -2, -1, 1, 2, 3]))
        else:
            new = rng.choice([m for m in MONTHS if m != tok])
        out = _sub_token(ans, tok, new)
        if out:
            return out, f"date:{tok}->{new}"
    return None


def flip_entity(ans: str, ctx: str, rng: random.Random):
    # Single capitalised words must repeat in the context (real subjects do; sentence-initial
    # common words mostly don't); multi-word runs are trusted as names.
    def solid(e: str, text: str) -> bool:
        return len(e.split()) > 1 or len(re.findall(r"(?<!\w)" + re.escape(e) + r"(?!\w)", text)) >= 2

    in_ans = sorted(e for e in set(_ENTITY_RE.findall(ans))
                    if e in ctx and e.split()[0] not in _ENTITY_STOP and len(e) > 3 and solid(e, ctx))
    pool = sorted(e for e in set(_ENTITY_RE.findall(ctx))
                  if e not in ans and e.split()[0] not in _ENTITY_STOP and len(e) > 3 and solid(e, ctx))
    rng.shuffle(in_ans)
    for tok in in_ans:
        same_shape = [p for p in pool if (len(p.split()) > 1) == (len(tok.split()) > 1)
                      and tok not in p and p not in tok]
        if not same_shape:
            continue
        out = _sub_token(ans, tok, rng.choice(same_shape))
        if out:
            return out, f"entity:{tok}"
    return None


def make_hard_negative(ex: dict, rng: random.Random) -> Optional[dict]:
    """Flip exactly one fact the source supports; returns a hallucinated copy or None."""
    flips: list[Callable[[], Optional[tuple]]] = [
        lambda: flip_number(ex["answer"], ex["source"], rng),
        lambda: flip_direction(ex["answer"], rng),
        lambda: flip_date(ex["answer"], ex["source"], rng),
        lambda: flip_entity(ex["answer"], ex["source"], rng),
    ]
    order = rng.sample(range(4), 4)
    order.sort(key=lambda i: [0.0, 0.5, 1.0, 1.5][i] + rng.random())  # bias toward number/direction
    for i in order:
        r = flips[i]()
        if r and r[0] != ex["answer"]:
            return {"question": ex["question"], "source": ex["source"], "answer": r[0],
                    "label": "hallucinated", "flip": r[1]}
    return None


# ----------------------------------------------------------------------------- build
@dataclass(frozen=True)
class DataConfig:
    seed: int
    ragtruth_train: int
    ragtruth_val: int
    ragtruth_test: int
    vitaminc_train: dict
    vitaminc_val: int
    vitaminc_test: int
    hard_neg_train: int
    hard_neg_test: int
    question_drop: float

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def build_splits(cfg: DataConfig, ragtruth, vitaminc) -> dict[str, list[dict]]:
    """Reproduce the v2 data mix. ``ragtruth``/``vitaminc`` are HF DatasetDicts.

    The order of RNG calls matches the original script exactly; do not reorder.
    """
    rng = random.Random(cfg.seed)

    rt_all = [e for row in ragtruth["train"] if (e := ragtruth_example(dict(row)))]
    rng.shuffle(rt_all)
    rt_test = [e for row in ragtruth["test"] if (e := ragtruth_example(dict(row)))][: cfg.ragtruth_test]
    rt_val, rt_pool = rt_all[: cfg.ragtruth_val], rt_all[cfg.ragtruth_val:]
    rt_train = rt_pool[: cfg.ragtruth_train]

    def drop_question(ex: dict) -> dict:
        e = dict(ex)
        if rng.random() < cfg.question_drop:
            e["question"] = ""
        return e

    rt_train = [drop_question(e) for e in rt_train]

    hn_train = []
    for ex in (e for e in rt_train if e["label"] == "grounded"):
        if len(hn_train) >= cfg.hard_neg_train:
            break
        hn = make_hard_negative(ex, rng)
        if hn:
            hn_train.append(hn)

    # Flipped holdout from grounded TEST rows: (original grounded, flipped hallucinated) pairs.
    hn_test, hn_test_orig = [], []
    for ex in (e for e in rt_test if e["label"] == "grounded"):
        if len(hn_test) >= cfg.hard_neg_test:
            break
        hn = make_hard_negative(ex, rng)
        if hn:
            hn_test.append(hn)
            hn_test_orig.append(ex)

    vc_target = cfg.vitaminc_train
    vc_by_label: dict[str, list[dict]] = {k: [] for k in vc_target}
    vc_train_split = vitaminc["train"]
    idx = list(range(len(vc_train_split)))
    rng.shuffle(idx)
    for i in idx:
        r = vc_train_split[i]
        if r["label"] in vc_by_label and len(vc_by_label[r["label"]]) < vc_target[r["label"]] + cfg.vitaminc_val:
            e = vitaminc_example(r)
            if e:
                vc_by_label[r["label"]].append(e)
        if all(len(v) >= vc_target[k] + cfg.vitaminc_val for k, v in vc_by_label.items()):
            break
    vc_train = sum((v[: vc_target[k]] for k, v in vc_by_label.items()), [])
    vc_val = sum((v[vc_target[k]: vc_target[k] + cfg.vitaminc_val // 3 + 1]
                  for k, v in vc_by_label.items()), [])[: cfg.vitaminc_val]
    test_split = "test" if "test" in vitaminc else "validation"
    vc_test_raw = vitaminc[test_split].shuffle(seed=cfg.seed).select(
        range(min(cfg.vitaminc_test * 2, len(vitaminc[test_split]))))
    vc_test = [e for r in vc_test_raw if (e := vitaminc_example(dict(r)))][: cfg.vitaminc_test]

    train = rt_train + vc_train + hn_train
    rng.shuffle(train)
    return {
        "train": train,
        "val": rt_val + vc_val,
        "ragtruth_test": rt_test,
        "vitaminc_test": vc_test,
        "flipped_test_hallucinated": hn_test,
        "flipped_test_original": hn_test_orig,
    }


def summarize(rows: list[dict]) -> dict:
    labels = Counter(r["label"] for r in rows)
    out = {"n": len(rows), "grounded": labels.get("grounded", 0), "hallucinated": labels.get("hallucinated", 0)}
    flips = Counter(r["flip"].split(":")[0] for r in rows if "flip" in r)
    if flips:
        out["flip_types"] = dict(sorted(flips.items()))
    return out


# The five hand-written cases v1 failed (2/5). Kept as a qualitative smoke check, not a metric.
MANUAL_CASES = [
    {"source": "The Amazon rainforest covers about 5.5 million square kilometers, with roughly 60% located in Brazil. It hosts an estimated 390 billion individual trees and around 16,000 species.",
     "answer": "Most of the Amazon rainforest, around 60%, lies within Brazil.", "label": "grounded"},
    {"source": "The Amazon rainforest covers about 5.5 million square kilometers, with roughly 60% located in Brazil. It hosts an estimated 390 billion individual trees and around 16,000 species.",
     "answer": "The Amazon rainforest produces 20% of the world's oxygen and covers 5.5 million square kilometers.", "label": "hallucinated"},
    {"source": "In Q3, the company's revenue increased 12% year-over-year to $2.1 billion, driven by strong cloud subscription growth.",
     "answer": "Revenue grew 12% in the third quarter, reaching $2.1 billion.", "label": "grounded"},
    {"source": "In Q3, the company's revenue increased 12% year-over-year to $2.1 billion, driven by strong cloud subscription growth.",
     "answer": "The company's revenue fell 12% in Q3.", "label": "hallucinated"},
    {"source": "The clinical trial found the vaccine was 94% effective at preventing symptomatic infection among adults aged 18-64.",
     "answer": "The vaccine showed 49% effectiveness in preventing symptomatic infection.", "label": "hallucinated"},
]

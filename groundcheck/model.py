"""The grounding detector.

``GroundCheck`` answers: *given the source/context an AI was supposed to use, is its
answer supported (grounded) or unsupported (hallucinated)?* It returns a
``grounded_score`` and a label.

Two backends behind one interface, always chosen explicitly:

* **model**     — the fine-tuned ModernBERT classifier, pinned to an exact Hugging Face
  revision. The default. If it cannot be loaded, construction raises
  :class:`ModelUnavailableError`; nothing is substituted.
* **heuristic** — a dependency-free lexical-overlap baseline. Only used when requested
  (``backend="heuristic"`` / ``GROUNDCHECK_BACKEND=heuristic``). Useful for tests and as
  the naive baseline row in the benchmark.

Every ``CheckResult`` records which backend produced it.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Optional

from .config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, DetectorSettings, get_detector_settings
from .schemas import CheckResult

# Small, fixed stop-word list for the heuristic backend (kept inline so the fallback has
# no third-party dependencies). Not linguistically exhaustive — just enough to stop
# function words from inflating the overlap score.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into over under
    is are was were be been being am do does did has have had will would shall should
    can could may might must this that these those it its they them their there here
    as not no nor so than too very s t re ve ll d m o you your we our i he she his her
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _content_tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens with stop-words removed."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


class ModelUnavailableError(RuntimeError):
    """The model backend was requested (the default) but the model could not be loaded."""


class InputTooLongError(ValueError):
    """The answer alone does not fit in ``max_length`` tokens.

    Only the source side is ever truncated; cutting the answer would mean judging a
    different claim than the one asked about. Split long answers and check them separately.
    """


EXPECTED_LABELS = {0: "grounded", 1: "hallucinated"}


def _resolved_hub_commit(path: str, revision: Optional[str], config) -> Optional[str]:
    """The Hub commit the weights were loaded from, or None for a local directory.

    transformers 4.x records it as ``config._commit_hash``; 5.x dropped that attribute, so
    fall back to the snapshot directory the Hub cache resolved ``config.json`` into
    (``.../snapshots/<commit>/config.json``), which works online and offline.
    """
    import os

    if os.path.isdir(path):
        return None
    commit = getattr(config, "_commit_hash", None)
    if commit:
        return commit
    from huggingface_hub import hf_hub_download

    try:
        cached = hf_hub_download(path, "config.json", revision=revision)
    except Exception:  # pragma: no cover - the model itself loaded, so this is not expected
        return None
    parts = os.path.normpath(cached).split(os.sep)
    if len(parts) >= 3 and parts[-3] == "snapshots":
        return parts[-2]
    return None


class GroundCheck:
    """Load once, call ``check`` (or ``check_many``) many times.

    Parameters
    ----------
    settings:
        Optional :class:`DetectorSettings`; defaults to environment-driven settings.
    **overrides:
        Any ``DetectorSettings`` field, e.g. ``GroundCheck(backend="heuristic")``.
    """

    def __init__(self, settings: Optional[DetectorSettings] = None, **overrides) -> None:
        base = settings or get_detector_settings()
        if "model_path" in overrides and "revision" not in overrides:
            # The pinned revision belongs to the published model only; a different model
            # path must not inherit it (it would name a commit that path doesn't have).
            overrides["revision"] = (DEFAULT_MODEL_REVISION
                                     if overrides["model_path"] == DEFAULT_MODEL_ID else None)
        self.settings = replace(base, **overrides) if overrides else base
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = "cpu"
        self._grounded_index = 0
        self._resolved_revision: Optional[str] = None
        self.backend = self._init_backend()

    # ------------------------------------------------------------------ #
    # backend selection / loading                                        #
    # ------------------------------------------------------------------ #
    def _init_backend(self) -> str:
        if self.settings.backend == "heuristic":
            return "heuristic"
        try:
            self._load_model()
        except ModelUnavailableError:
            raise
        except ImportError as exc:
            raise ModelUnavailableError(
                f"The model backend needs torch and transformers ({exc}). Install them with "
                f"`pip install \"groundcheck-rag[model]\"`, or pass backend='heuristic' to use "
                f"the lexical-overlap baseline knowingly."
            ) from exc
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load model {self.settings.model_path!r} "
                f"(revision {self.settings.revision or 'unpinned'}): {exc}"
            ) from exc
        return "model"

    def _load_model(self) -> None:
        """Import torch/transformers lazily and load the pinned classifier."""
        import torch  # noqa: WPS433 (lazy import is intentional)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        path = self.settings.model_path
        revision = self.settings.revision
        self._tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(path, revision=revision)
        self._model.eval()
        self._resolved_revision = _resolved_hub_commit(path, revision, self._model.config)

        # The label order is part of the model contract: refuse anything else rather than
        # guess, so a model with flipped labels can never silently invert every verdict.
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        if id2label != EXPECTED_LABELS:
            raise ModelUnavailableError(
                f"Model {path!r} has labels {id2label}; GroundCheck expects {EXPECTED_LABELS}"
            )
        self._grounded_index = 0

        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._device = device

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def model_info(self) -> dict:
        """What is answering, and exactly which weights (for logs and provenance)."""
        if self.backend == "heuristic":
            return {"backend": "heuristic"}
        return {
            "backend": "model",
            "path": self.settings.model_path,
            "requested_revision": self.settings.revision,
            "resolved_revision": self._resolved_revision,
            "architecture": type(self._model).__name__,
            "max_length": self.settings.max_length,
            "device": self._device,
        }

    def check(self, source: str, answer: str, question: Optional[str] = None) -> CheckResult:
        """Check a single answer against its source. ``question`` is optional context."""
        t0 = time.perf_counter()
        score = (
            self._score_model(source, answer, question)
            if self.backend == "model"
            else self._score_heuristic(source, answer)
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        grounded = score >= self.settings.threshold
        return CheckResult(
            label="grounded" if grounded else "hallucinated",
            grounded=grounded,
            grounded_score=round(float(score), 4),
            backend=self.backend,
            latency_ms=round(latency_ms, 2),
        )

    def check_many(
        self,
        items: list[dict],
        batch_size: int = 16,
    ) -> list[CheckResult]:
        """Batch version. Each item is a dict with keys: source, answer, question?."""
        if self.backend != "model":
            return [self.check(it["source"], it["answer"], it.get("question")) for it in items]
        return self._score_model_batch(items, batch_size)

    # ------------------------------------------------------------------ #
    # model backend                                                      #
    # ------------------------------------------------------------------ #
    # Room the answer must leave for special tokens and at least a little source text.
    _MIN_SOURCE_TOKENS = 16

    def _check_answer_fits(self, answers: list[str]) -> None:
        budget = self.settings.max_length - 3 - self._MIN_SOURCE_TOKENS
        for i, ids in enumerate(self._tokenizer(answers, add_special_tokens=False)["input_ids"]):
            if len(ids) > budget:
                raise InputTooLongError(
                    f"answer{'' if len(answers) == 1 else f' #{i}'} is {len(ids)} tokens; at "
                    f"max_length={self.settings.max_length} it may be at most {budget}. "
                    f"Split it into shorter claims."
                )

    def _encode(self, source: str, answer: str, question: Optional[str]):
        self._check_answer_fits([answer])
        premise = source if not question else f"{question}\n\n{source}"
        return self._tokenizer(
            premise,
            answer,
            truncation="only_first",
            max_length=self.settings.max_length,
            return_tensors="pt",
        )

    def _score_model(self, source: str, answer: str, question: Optional[str]) -> float:
        torch = self._torch
        enc = self._encode(source, answer, question).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
        return probs[self._grounded_index].item()

    def _score_model_batch(self, items: list[dict], batch_size: int) -> list[CheckResult]:
        torch = self._torch
        out: list[CheckResult] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            premises = [
                it["source"] if not it.get("question") else f"{it['question']}\n\n{it['source']}"
                for it in chunk
            ]
            answers = [it["answer"] for it in chunk]
            self._check_answer_fits(answers)
            t0 = time.perf_counter()
            enc = self._tokenizer(
                premises,
                answers,
                truncation="only_first",
                max_length=self.settings.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits, dim=-1)
            per_item_ms = (time.perf_counter() - t0) * 1000.0 / max(len(chunk), 1)
            for p in probs:
                score = p[self._grounded_index].item()
                grounded = score >= self.settings.threshold
                out.append(
                    CheckResult(
                        label="grounded" if grounded else "hallucinated",
                        grounded=grounded,
                        grounded_score=round(float(score), 4),
                        backend="model",
                        latency_ms=round(per_item_ms, 2),
                    )
                )
        return out

    # ------------------------------------------------------------------ #
    # heuristic backend (dependency-free fallback + naive baseline)      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score_heuristic(source: str, answer: str) -> float:
        """Fraction of the answer's content tokens that also appear in the source.

        Simple, transparent, and deterministic: if an answer introduces many tokens the
        source never mentions, it is more likely to be unsupported. This is intentionally
        weak — it exists to keep everything runnable without a GPU and to serve as the
        floor the trained model must clear.
        """
        ans_tokens = _content_tokens(answer)
        if not ans_tokens:
            return 1.0  # nothing asserted => nothing to hallucinate
        src_tokens = set(_content_tokens(source))
        supported = sum(1 for t in ans_tokens if t in src_tokens)
        return supported / len(ans_tokens)

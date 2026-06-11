"""The grounding detector.

``GroundCheck`` answers: *given the source/context an AI was supposed to use, is its
answer supported (grounded) or unsupported (hallucinated)?* It returns a calibrated
``grounded_score`` and a label.

Two backends behind one interface:

* **model**     — the fine-tuned ModernBERT classifier (the real product). Loaded lazily
  so importing this package costs nothing until you actually run a check.
* **heuristic** — a dependency-free lexical-overlap baseline. Used automatically when no
  trained model is available, so the API, web demo, tests and CI run with zero setup and
  zero heavy dependencies. It is also reported as the "naive baseline" row in the
  benchmark — a trained model that can't beat token-overlap isn't worth shipping.

The backend that actually answered is recorded on every ``CheckResult`` so nothing is
ever silently faked.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from .config import DetectorSettings, get_detector_settings
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


class GroundCheck:
    """Load once, call ``check`` (or ``check_many``) many times.

    Parameters
    ----------
    settings:
        Optional :class:`DetectorSettings`; defaults to environment-driven settings.
    """

    def __init__(self, settings: Optional[DetectorSettings] = None) -> None:
        self.settings = settings or get_detector_settings()
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._grounded_index = 0
        self.backend = self._init_backend()

    # ------------------------------------------------------------------ #
    # backend selection / loading                                        #
    # ------------------------------------------------------------------ #
    def _init_backend(self) -> str:
        want = self.settings.backend
        if want == "heuristic":
            return "heuristic"
        if want in {"auto", "model"}:
            if not self.settings.model_path and want == "auto":
                # Nothing trained yet — fall back silently (this is the expected pre-train state).
                return "heuristic"
            try:
                self._load_model()
                return "model"
            except Exception as exc:  # pragma: no cover - depends on optional heavy deps
                if want == "model":
                    raise RuntimeError(
                        f"GROUNDCHECK_BACKEND=model but the model could not be loaded "
                        f"({exc}). Train one (see train/) or set the backend to 'auto'."
                    ) from exc
                return "heuristic"
        raise ValueError(f"Unknown backend: {want!r}")

    def _load_model(self) -> None:
        """Lazily import torch/transformers and load the fine-tuned classifier."""
        import torch  # noqa: WPS433 (lazy import is intentional)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        path = self.settings.model_path
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._model.eval()

        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._device = device

        # Find which output index means "grounded/supported" from the model config,
        # so the train-time label order can never silently flip the score.
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self._grounded_index = next(
            (i for i, lbl in id2label.items() if lbl.startswith("ground") or "support" in lbl),
            0,
        )

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
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
    def _encode(self, source: str, answer: str, question: Optional[str]):
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

"""Environment-driven settings.

Two independent concerns live here:

* the **detector** — which model to load (pinned to an exact Hugging Face revision by
  default), which backend answers, the decision threshold, input length and device.
  The backend is always explicit: ``model`` (the default) fails loudly if the model
  cannot be loaded; ``heuristic`` is a lexical-overlap baseline that must be asked for.
* the **judge baseline** — an OpenAI-compatible LLM used only by the benchmark. The same
  LLM_* variables work for Groq, OpenAI, OpenRouter, or a local Ollama; leave the key
  empty to run the judge in deterministic mock mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:  # optional: the package runs fine without a .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


# Base model the published classifier was fine-tuned from (see training/configs/v2.json).
DEFAULT_BASE_MODEL = "answerdotai/ModernBERT-base"

# The published GroundCheck v2 model, pinned to an immutable Hub commit. Weights were
# uploaded in 0c7dd063; 998cec35 only changed the model card, so both resolve to the same
# weight files. Pinning means a later push to the Hub can never silently change results.
DEFAULT_MODEL_ID = "Pranshurs/groundcheck-modernbert"
DEFAULT_MODEL_REVISION = "998cec35563d6b90947409d1c7510adac7f7c80c"

BACKENDS = ("model", "heuristic")

# Token budget for (question + source, answer). The model was fine-tuned and its published
# numbers measured at 512; at 2048 it scored higher on the same RAGTruth rows (F1 0.6955 vs
# 0.6817, paired bootstrap ΔF1 95% CI [-0.001, +0.028]) at ~2.5x the CPU time, and 2048 is
# what 0.1.0 used, so it stays the default. See eval/reports/ and the README.
DEFAULT_MAX_LENGTH = 2048
PUBLISHED_PROTOCOL_MAX_LENGTH = 512


@dataclass
class DetectorSettings:
    model_path: str = DEFAULT_MODEL_ID   # local dir or HF id of the fine-tuned classifier
    backend: str = "model"               # "model" | "heuristic" — never chosen implicitly
    threshold: float = 0.5               # grounded if grounded_score >= threshold
    max_length: int = DEFAULT_MAX_LENGTH  # token cap for (question + source, answer)
    device: str = "auto"                 # "auto" | "cpu" | "cuda"
    revision: Optional[str] = DEFAULT_MODEL_REVISION  # HF commit; ignored for local dirs

    def __post_init__(self) -> None:
        self.backend = (self.backend or "").strip().lower()
        if self.backend == "auto":
            raise ValueError(
                "GROUNDCHECK_BACKEND=auto was removed in 0.2.0: it silently answered with the "
                "lexical heuristic whenever the model failed to load. Use 'model' (default) or "
                "request 'heuristic' explicitly."
            )
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend {self.backend!r}; expected one of {BACKENDS}")
        if self.backend == "model" and not self.model_path:
            raise ValueError("backend='model' needs a model_path (HF id or local directory)")


@dataclass
class JudgeSettings:
    mode: str           # "live" or "mock"
    base_url: str
    api_key: str
    model: str
    temperature: float
    # $ per 1M tokens, used by the cost model in the benchmark (override per provider).
    price_in_per_mtok: float
    price_out_per_mtok: float


def get_detector_settings() -> DetectorSettings:
    model_path = os.getenv("GROUNDCHECK_MODEL_PATH", "").strip() or DEFAULT_MODEL_ID
    revision = os.getenv("GROUNDCHECK_MODEL_REVISION", "").strip()
    if not revision:
        # The pinned revision only applies to the published model; a custom model path
        # resolves to whatever that path contains unless a revision is given.
        revision = DEFAULT_MODEL_REVISION if model_path == DEFAULT_MODEL_ID else None
    return DetectorSettings(
        model_path=model_path,
        backend=os.getenv("GROUNDCHECK_BACKEND", "model"),
        threshold=float(os.getenv("GROUNDCHECK_THRESHOLD", "0.5")),
        max_length=int(os.getenv("GROUNDCHECK_MAX_LENGTH", str(DEFAULT_MAX_LENGTH))),
        device=os.getenv("GROUNDCHECK_DEVICE", "auto").strip().lower(),
        revision=revision,
    )


def get_judge_settings() -> JudgeSettings:
    """Build judge settings, auto-selecting mock mode when no API key is present."""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    mode = os.getenv("LLM_MODE", "").strip().lower()
    if mode not in {"live", "mock"}:
        mode = "live" if api_key else "mock"

    return JudgeSettings(
        mode=mode,
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip(),
        api_key=api_key,
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip(),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        price_in_per_mtok=float(os.getenv("JUDGE_PRICE_IN_PER_MTOK", "0.59")),
        price_out_per_mtok=float(os.getenv("JUDGE_PRICE_OUT_PER_MTOK", "0.79")),
    )

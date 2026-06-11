"""Environment-driven settings.

Two independent concerns live here:

* the **detector** — which fine-tuned model to load, the decision threshold, max input
  length, and device. If no trained model is found, the detector falls back to a
  dependency-free lexical-overlap heuristic so the API, demo, tests and CI all run with
  zero setup (same philosophy as the offline mock mode in JobHunt Copilot).
* the **judge baseline** — an OpenAI-compatible LLM used only by the benchmark. The same
  LLM_* variables work for Groq, OpenAI, OpenRouter, or a local Ollama; leave the key
  empty to run the judge in deterministic mock mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional: the package runs fine without a .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


# Default training/inference backbone. ModernBERT handles RAGTruth's long contexts
# (mean ~800 tokens, max ~2,600) where a 512-token encoder would truncate half the docs.
DEFAULT_BASE_MODEL = "answerdotai/ModernBERT-base"

# Published GroundCheck model on the Hugging Face Hub — loaded by default so
# `GroundCheck()` works out of the box (override with GROUNDCHECK_MODEL_PATH).
DEFAULT_MODEL_ID = "Pranshurs/groundcheck-modernbert"


@dataclass
class DetectorSettings:
    model_path: str     # local dir or HF id of the fine-tuned classifier ("" => heuristic)
    backend: str        # "auto" | "model" | "heuristic"
    threshold: float    # grounded if grounded_score >= threshold
    max_length: int     # token cap for (context + question + answer)
    device: str         # "auto" | "cpu" | "cuda"


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
    return DetectorSettings(
        model_path=os.getenv("GROUNDCHECK_MODEL_PATH", DEFAULT_MODEL_ID).strip(),
        backend=os.getenv("GROUNDCHECK_BACKEND", "auto").strip().lower(),
        threshold=float(os.getenv("GROUNDCHECK_THRESHOLD", "0.5")),
        max_length=int(os.getenv("GROUNDCHECK_MAX_LENGTH", "2048")),
        device=os.getenv("GROUNDCHECK_DEVICE", "auto").strip().lower(),
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

"""GroundCheck — a small, cheap hallucination / grounding detector.

Typical use::

    from groundcheck import GroundCheck

    gc = GroundCheck()                       # loads the model, or the heuristic fallback
    r = gc.check(source="The capital of France is Paris.",
                 answer="Paris is the capital of France.")
    print(r.label, r.grounded_score)         # -> grounded 0.97

Point it at a fine-tuned model with the GROUNDCHECK_MODEL_PATH env var (a local dir or a
Hugging Face id). With nothing trained yet, it transparently uses a lexical-overlap
baseline and says so via ``r.backend``.
"""

from __future__ import annotations

from typing import Optional

from .model import GroundCheck
from .schemas import CheckResult

__all__ = ["GroundCheck", "CheckResult", "check", "__version__"]
__version__ = "0.1.0"

_DEFAULT: Optional[GroundCheck] = None


def check(source: str, answer: str, question: Optional[str] = None) -> CheckResult:
    """One-shot convenience wrapper that lazily builds and reuses a default detector."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GroundCheck()
    return _DEFAULT.check(source, answer, question)

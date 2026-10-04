"""GroundCheck — a small grounding / hallucination detector for RAG answers.

Typical use::

    from groundcheck import GroundCheck

    gc = GroundCheck()                       # loads the pinned published model, or raises
    r = gc.check(source="The capital of France is Paris.",
                 answer="Paris is the capital of France.")
    print(r.label, r.grounded_score, r.backend)

``GroundCheck(backend="heuristic")`` gives the dependency-free lexical-overlap baseline;
it is never used unless asked for.
"""

from __future__ import annotations

from typing import Optional

from .model import GroundCheck, InputTooLongError, ModelUnavailableError
from .schemas import CheckResult

__all__ = ["GroundCheck", "CheckResult", "InputTooLongError", "ModelUnavailableError", "check", "__version__"]
__version__ = "0.2.0"

_DEFAULT: Optional[GroundCheck] = None


def check(source: str, answer: str, question: Optional[str] = None) -> CheckResult:
    """One-shot convenience wrapper that lazily builds and reuses a default detector."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GroundCheck()
    return _DEFAULT.check(source, answer, question)

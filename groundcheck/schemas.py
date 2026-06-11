"""Typed results for a grounding check.

A single check answers one question: *is this answer supported by the source it was
given?* The result carries a calibrated probability (``grounded_score``), the derived
label, which backend produced it, and how long it took — the last two matter because the
whole point of this project is the accuracy/latency/cost trade-off versus an LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class CheckResult:
    """The outcome of checking one (source, answer) pair."""

    label: str            # "grounded" or "hallucinated"
    grounded: bool        # convenience: label == "grounded"
    grounded_score: float # P(answer is supported by source), in [0, 1]
    backend: str          # "model" (fine-tuned encoder) or "heuristic" (overlap fallback)
    latency_ms: float     # wall-clock time for this single check

    @property
    def hallucination_score(self) -> float:
        """P(answer is NOT supported) — the number you'd alert/threshold on."""
        return round(1.0 - self.grounded_score, 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hallucination_score"] = self.hallucination_score
        return d

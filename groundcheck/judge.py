"""LLM-as-a-judge baseline — the thing GroundCheck has to beat on cost.

This is *only* used by the benchmark. It wraps any OpenAI-compatible endpoint (Groq,
OpenAI, OpenRouter, local Ollama) behind the same ``LLM_*`` env vars as JobHunt Copilot,
and falls back to a deterministic mock so the benchmark runs offline.

Each call reports latency and token usage, so the benchmark can compute a real
$-per-1,000-checks figure rather than hand-waving about "cheaper".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .config import JudgeSettings, get_judge_settings

JUDGE_SYSTEM = (
    "You are a strict grounding checker. You are given a SOURCE and an ANSWER. "
    "Decide whether every claim in the ANSWER is directly supported by the SOURCE. "
    "If the ANSWER adds facts not in the SOURCE, or contradicts it, it is NOT grounded. "
    'Reply with ONLY a JSON object: {"grounded": true|false, "confidence": 0.0-1.0}. '
    "confidence is your probability that the ANSWER is fully supported by the SOURCE."
)


@dataclass
class JudgeResult:
    grounded: bool
    grounded_score: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class BaseJudge:
    def check(self, source: str, answer: str, question: Optional[str] = None) -> JudgeResult:
        raise NotImplementedError


def _build_user_prompt(source: str, answer: str, question: Optional[str]) -> str:
    parts = []
    if question:
        parts.append(f"QUESTION:\n{question}\n")
    parts.append(f"SOURCE:\n{source}\n")
    parts.append(f"ANSWER:\n{answer}\n")
    parts.append('Respond with only the JSON object: {"grounded": ..., "confidence": ...}')
    return "\n".join(parts)


class LiveJudge(BaseJudge):
    """Real LLM judge over an OpenAI-compatible API."""

    def __init__(self, settings: JudgeSettings) -> None:
        from openai import OpenAI  # lazy import: only needed in live mode

        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key or "x")
        self.model = settings.model
        self.temperature = settings.temperature

    def check(self, source: str, answer: str, question: Optional[str] = None) -> JudgeResult:
        import time

        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _build_user_prompt(source, answer, question)},
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        content = resp.choices[0].message.content or ""
        grounded, score = _parse_judgement(content)

        usage = getattr(resp, "usage", None)
        return JudgeResult(
            grounded=grounded,
            grounded_score=score,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class MockJudge(BaseJudge):
    """Deterministic offline judge so the benchmark runs with no key.

    It is a weak stand-in (overlap-with-threshold), present only to exercise the harness.
    Real headline numbers must come from a LIVE judge — the benchmark labels which judge
    produced the results so a mock run can never be mistaken for a real comparison.
    """

    def __init__(self, settings: Optional[JudgeSettings] = None) -> None:
        self.settings = settings  # unused; kept so build_judge() can pass settings uniformly

    def check(self, source: str, answer: str, question: Optional[str] = None) -> JudgeResult:
        import time

        from .model import _content_tokens

        t0 = time.perf_counter()
        ans = _content_tokens(answer)
        src = set(_content_tokens(source))
        frac = 1.0 if not ans else sum(1 for t in ans if t in src) / len(ans)
        # Squash toward a decision so the mock behaves a little more like a confident judge.
        score = round(min(1.0, max(0.0, (frac - 0.4) / 0.4)), 4)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        approx = lambda s: max(1, len(s) // 4)  # ~4 chars/token
        return JudgeResult(
            grounded=score >= 0.5,
            grounded_score=score,
            latency_ms=latency_ms,
            prompt_tokens=approx(JUDGE_SYSTEM) + approx(source) + approx(answer),
            completion_tokens=12,
        )


def _parse_judgement(text: str) -> tuple[bool, float]:
    """Pull {grounded, confidence} out of a model reply, tolerating stray prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            grounded = bool(obj.get("grounded"))
            conf = float(obj.get("confidence", 1.0 if grounded else 0.0))
            conf = min(1.0, max(0.0, conf))
            # Express as P(grounded) regardless of which way the judge leaned.
            return grounded, conf if grounded else 1.0 - conf
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback: look for a bare yes/no.
    lowered = text.lower()
    grounded = "true" in lowered or "grounded" in lowered and "not grounded" not in lowered
    return grounded, 1.0 if grounded else 0.0


def build_judge(settings: Optional[JudgeSettings] = None) -> BaseJudge:
    settings = settings or get_judge_settings()
    return LiveJudge(settings) if settings.mode == "live" else MockJudge(settings)

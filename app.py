"""FastAPI server — the GroundCheck inference API and live demo.

Run locally:
    uvicorn app:app --reload
    # open http://localhost:8000

Endpoints:
    GET  /            single-page demo (paste source + answer -> grounded? + score)
    GET  /api/health  {status, backend, threshold}
    POST /api/check   {source, answer, question?} -> CheckResult
    POST /api/check_batch {items:[{source, answer, question?}, ...]} -> [CheckResult]

The detector is loaded once at startup. With no trained model present it runs on the
heuristic fallback and says so via ``backend`` — the service always boots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from groundcheck.config import get_detector_settings
from groundcheck.model import GroundCheck

ROOT = Path(__file__).parent
app = FastAPI(title="GroundCheck", version="0.1.0")

# Load the detector once (model or heuristic fallback).
_SETTINGS = get_detector_settings()
_DETECTOR = GroundCheck(_SETTINGS)


class CheckRequest(BaseModel):
    source: str
    answer: str
    question: Optional[str] = None


class BatchItem(BaseModel):
    source: str
    answer: str
    question: Optional[str] = None


class BatchRequest(BaseModel):
    items: list[BatchItem]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": _DETECTOR.backend,
        "threshold": _SETTINGS.threshold,
        "model_path": _SETTINGS.model_path or None,
    }


@app.post("/api/check")
def check(req: CheckRequest) -> JSONResponse:
    result = _DETECTOR.check(req.source, req.answer, req.question)
    return JSONResponse(result.to_dict())


@app.post("/api/check_batch")
def check_batch(req: BatchRequest) -> JSONResponse:
    items = [it.model_dump() for it in req.items]
    results = _DETECTOR.check_many(items)
    return JSONResponse([r.to_dict() for r in results])

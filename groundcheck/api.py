"""FastAPI server — the GroundCheck inference API and demo page.

    pip install "groundcheck-rag[serve,model]"
    uvicorn groundcheck.api:app --port 8000

Endpoints:
    GET  /                 single-page demo (paste source + answer -> grounded? + score)
    GET  /api/health       {status, backend, threshold, model}
    POST /api/check        {source, answer, question?} -> CheckResult
    POST /api/check_batch  {items: [{source, answer, question?}, ...]} -> [CheckResult]

The detector is built once at startup from environment settings (see ``config.py``).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from importlib import resources
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import get_detector_settings
from .model import GroundCheck, InputTooLongError

# Request limits. The model truncates long sources anyway; these exist so one request
# cannot pin the CPU or the process memory.
MAX_SOURCE_CHARS = int(os.getenv("GROUNDCHECK_MAX_SOURCE_CHARS", "100000"))
MAX_ANSWER_CHARS = int(os.getenv("GROUNDCHECK_MAX_ANSWER_CHARS", "20000"))
MAX_QUESTION_CHARS = int(os.getenv("GROUNDCHECK_MAX_QUESTION_CHARS", "4000"))
MAX_BATCH_ITEMS = int(os.getenv("GROUNDCHECK_MAX_BATCH_ITEMS", "64"))

_state: dict = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_detector_settings()
    _state["settings"] = settings
    _state["detector"] = GroundCheck(settings)
    yield
    _state.clear()


app = FastAPI(title="GroundCheck", version=__version__, lifespan=_lifespan)


@app.exception_handler(InputTooLongError)
async def _too_long(_: Request, exc: InputTooLongError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


class CheckRequest(BaseModel):
    source: str = Field(min_length=1, max_length=MAX_SOURCE_CHARS)
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)
    question: Optional[str] = Field(default=None, max_length=MAX_QUESTION_CHARS)


class BatchRequest(BaseModel):
    items: list[CheckRequest] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return resources.files("groundcheck").joinpath("templates/index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    detector: GroundCheck = _state["detector"]
    return {
        "status": "ok",
        "version": __version__,
        "backend": detector.backend,
        "threshold": _state["settings"].threshold,
        "model": detector.model_info(),
    }


@app.post("/api/check")
def check(req: CheckRequest) -> JSONResponse:
    result = _state["detector"].check(req.source, req.answer, req.question)
    return JSONResponse(result.to_dict())


@app.post("/api/check_batch")
def check_batch(req: BatchRequest) -> JSONResponse:
    items = [it.model_dump() for it in req.items]
    results = _state["detector"].check_many(items)
    return JSONResponse([r.to_dict() for r in results])

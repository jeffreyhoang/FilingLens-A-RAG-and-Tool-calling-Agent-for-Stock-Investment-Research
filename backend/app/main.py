"""FilingLens API. Run from backend/: uv run uvicorn app.main:app --reload

GET  /health      the app is up
GET  /health/db   Supabase is reachable
POST /chat        answer a question with optional chat history
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import openai
import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from app import pipeline
from app.config import get_settings
from app.steps.standalone import Message
from app.steps.tools import metrics

MAX_QUESTION_CHARS = 1000
MAX_HISTORY = 20  # earlier messages accepted; the rewrite uses the last few
CORS_ORIGINS = ["http://localhost:3000"]  # the local frontend


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    pipeline.search_index()  # cached; connects to Supabase on the first search
    metrics()  # read from Supabase before the first request
    yield


app = FastAPI(title="FilingLens", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS
        ),
    ]
    history: list[Message] = Field(default=[], max_length=MAX_HISTORY)


@app.exception_handler(RequestValidationError)
async def bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=400, content={"error": "bad request", "details": errors}
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/db")
def health_db() -> dict[str, str | None]:
    with psycopg.connect(get_settings().database_url, connect_timeout=10) as conn:
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return {"status": "ok", "pgvector": row[0] if row else None}


@app.post("/chat", response_model=pipeline.Response)
def chat(request: ChatRequest) -> pipeline.Response | JSONResponse:
    """Blocking: FastAPI runs it in a worker thread."""
    try:
        return pipeline.answer(request.question, request.history)
    except (openai.APIError, RuntimeError) as exc:  # model call failed or empty
        return JSONResponse(
            status_code=502,
            content={"error": "model call failed", "detail": type(exc).__name__},
        )
    except psycopg.OperationalError as exc:  # Supabase unreachable or paused
        return JSONResponse(
            status_code=503,
            content={"error": "database unavailable", "detail": type(exc).__name__},
        )

"""FastAPI app: POST /search, GET /dinos and the static front-end."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.search.ranker import Ranker, Result, build_ranker, load_config

STATIC = Path(__file__).resolve().parents[1] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and embed the base once (cached on disk across restarts)."""
    app.state.ranker = build_ranker(load_config())
    yield


app = FastAPI(title="Dino Search", lifespan=lifespan)


class SearchRequest(BaseModel):
    """Body of POST /search."""

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=30)


def _ranker(request: Request) -> Ranker:
    """The Ranker built at startup."""
    return request.app.state.ranker


@app.post("/search")
def search(req: SearchRequest, request: Request) -> list[Result]:
    """Rank dinos by similarity to a free-text physical description."""
    return _ranker(request).search(req.query, req.top_k)


@app.get("/dinos")
def dinos(request: Request) -> list[dict]:
    """The full dataset as stored in data/dinos.json."""
    return [d.raw for d in _ranker(request).dinos]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The single-page front-end."""
    return FileResponse(STATIC / "index.html")

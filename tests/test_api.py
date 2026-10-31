"""End-to-end API check with the real model (slow: loads embeddings once)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_search_returns_ranked_explained_hits(client: TestClient) -> None:
    r = client.post("/search", json={"query": "três chifres e um escudo no pescoço", "top_k": 3})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 3 and hits[0]["nome"] == "Triceratops"
    assert "chifres" in hits[0]["tracos_correspondentes"]
    assert hits[0]["score_final"] >= hits[1]["score_final"]
    assert 0 <= hits[0]["similaridade"] <= 1


def test_search_validates_input(client: TestClient) -> None:
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422


def test_dinos_and_index(client: TestClient) -> None:
    assert len(client.get("/dinos").json()) == 30
    assert client.get("/").status_code == 200

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


def test_search_distinguishes_nasal_plates_from_back_plates(client: TestClient) -> None:
    response = client.post("/search", json={
        "query": "ceratopsiano sem chifres com uma placa grossa no nariz",
        "top_k": 1,
    })
    assert response.status_code == 200
    hit = response.json()[0]
    assert hit["nome"] == "Pachyrhinosaurus"
    assert set(hit["tracos_correspondentes"]) == {"ceratopsiano", "placa_nasal", "sem chifres"}


def test_reported_similarity_uses_raw_cosine_without_positive_attributes(client: TestClient) -> None:
    for query in ("qual é a capital da França?", "sem penas"):
        response = client.post("/search", json={"query": query, "top_k": 1})
        assert response.status_code == 200
        hit = response.json()[0]
        assert hit["penalidade"] == 0
        assert 0 < hit["similaridade"] == hit["score_embedding"] < 1


def test_search_validates_input(client: TestClient) -> None:
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422


def test_dinos_and_index(client: TestClient) -> None:
    assert len(client.get("/dinos").json()) == 64
    assert client.get("/").status_code == 200
    assert client.get("/all").status_code == 200


def test_every_dino_has_an_illustration_with_credits(client: TestClient) -> None:
    credits = client.get("/img/credits.json").json()
    for d in client.get("/dinos").json():
        r = client.get(f"/img/{d['nome']}.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", d["nome"]
        assert credits[d["nome"]]["autor"] and credits[d["nome"]]["licenca"].startswith("CC")

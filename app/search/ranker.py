"""Hybrid ranking: dense similarity + attribute match - negation penalty."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import re

import numpy as np
import yaml

from app.search.dinos import SIZES, Dino, load_dinos
from app.search.embedder import Aggregation, DenseIndex, Embedder
from app.search.extractor import Extractor, QueryAttrs, get_extractor, normalize

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

UNCERTAIN_WEIGHT = 0.75     # a debated trait counts as a partial match
UNCERTAIN_PENALTY = 0.5     # negating a debated trait costs half

# Partial compatibility between categorical values (symmetric).
DIET_COMPAT = {frozenset({"carnívoro", "piscívoro"}): 0.5, frozenset({"onívoro", "carnívoro"}): 0.5,
               frozenset({"onívoro", "herbívoro"}): 0.5}
POSTURE_COMPAT = {frozenset({"facultativo", "bípede"}): 0.5, frozenset({"facultativo", "quadrúpede"}): 0.5}

# Nicknames rewritten to the dino's name before name matching (patterns run on normalized words).
ALIASES = {r"\b(t ?)?rex\b|\btiranossauro\b": "tyrannosaurus",
           r"\bargentina\b": "argentinosaurus"}

# The best-known dinos get a small bonus: a vague query most likely means one of them.
POPULAR = {"Tyrannosaurus", "Spinosaurus aegyptiacus", "Triceratops", "Brachiosaurus", "Stegosaurus",
           "Ankylosaurus", "Parasaurolophus"}
POPULARITY_BONUS = 0.1


@dataclass(frozen=True)
class Weights:
    """score_final = alpha * dense + beta * attributes - gamma * penalty."""

    alpha: float
    beta: float
    gamma: float


@dataclass
class AttrMatch:
    """Attribute comparison between a query and one dino."""

    score: float = 0.0
    penalty: float = 0.0
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


@dataclass
class Result:
    """One ranked dino with its scores and per-trait explanation (also the API schema)."""

    nome: str
    score_final: float
    similaridade: float  # score_final over the best score this query could reach, in [0, 1]
    score_embedding: float
    score_atributos: float
    penalidade: float
    tracos_correspondentes: list[str]
    tracos_nao_correspondentes: list[str]
    tracos_conflitantes: list[str]
    descricao_fisica: str


def _categorical(q: str, d: str, compat: dict[frozenset[str], float]) -> float:
    return 1.0 if q == d else compat.get(frozenset({q, d}), 0.0)


def name_match(query: str, dino: Dino) -> float:
    """Fraction of the dino's name words present in the query ("spinosaurus" -> 0.5 for each species)."""
    text = " ".join(re.findall(r"\w+", normalize(query)))
    for pattern, name in ALIASES.items():
        text = re.sub(pattern, name, text)
    words = set(text.split())
    parts = normalize(dino.nome).split()
    return sum(p in words for p in parts) / len(parts)


def score_attributes(q: QueryAttrs, dino: Dino, name: float | None = None) -> AttrMatch:
    """Mean match over every attribute the query mentions; negated traits add penalty.
    `name` is this dino's name match, passed only when the query names some dino."""
    m = AttrMatch()
    parts: list[float] = []

    def record(label: str, s: float, conflict: bool) -> None:
        parts.append(s)
        if conflict:
            m.conflicts.append(label)
        elif s >= 1:
            m.matched.append(label)
        elif s > 0:
            m.matched.append(f"{label} (parcial)")
        else:
            m.missing.append(label)

    if q.postura:
        s = _categorical(q.postura, dino.postura, POSTURE_COMPAT)
        record(f"postura: {q.postura}", s, s == 0)
    if q.dieta:
        s = _categorical(q.dieta, dino.dieta, DIET_COMPAT)
        record(f"dieta: {q.dieta}", s, s == 0)
    if q.tamanho:
        dist = abs(SIZES.index(q.tamanho) - SIZES.index(dino.tamanho))
        record(f"tamanho: {q.tamanho}", 1 - dist / 3, dist >= 2)
    for t in sorted(q.tracos):
        record(t, (UNCERTAIN_WEIGHT if dino.tracos[t] else 1.0) if t in dino.tracos else 0.0, False)
    if name is not None:
        record(f"nome: {dino.nome}", name, False)
    for t in sorted(q.negados):
        if t in dino.tracos:
            m.penalty += UNCERTAIN_PENALTY if dino.tracos[t] else 1.0
            m.conflicts.append(f"sem {t}")
        else:
            m.matched.append(f"sem {t}")
    m.score = float(np.mean(parts)) if parts else 0.0
    return m


class Ranker:
    """Scores every dino for a query and returns the top-k with explanations."""

    def __init__(self, dinos: list[Dino], index: DenseIndex | None, extractor: Extractor,
                 weights: Weights, aggregation: Aggregation = "top2_mean") -> None:
        self.dinos = dinos
        self.index = index
        self.extractor = extractor
        self.weights = weights
        self.aggregation = aggregation

    def dense_scores(self, query: str) -> np.ndarray:
        """Raw cosine per dino (zeros when no index is loaded)."""
        if self.index is None:
            return np.zeros(len(self.dinos), dtype=np.float32)
        qv = self.index.embedder.encode_queries([query])[0]
        return self.index.scores(qv, self.aggregation)

    def search(self, query: str, top_k: int = 5, weights: Weights | None = None,
               dense: np.ndarray | None = None) -> list[Result]:
        """Rank dinos. `dense` lets callers reuse precomputed cosine scores."""
        w = weights or self.weights
        raw = self.dense_scores(query) if dense is None else dense
        # Min-max per query so dense and attribute scores share a [0, 1] scale;
        # E5 cosines sit in a narrow ~0.75-0.9 band otherwise.
        span = float(raw.max() - raw.min())
        dense_n = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
        attrs = self.extractor(query)
        names = [name_match(query, d) for d in self.dinos]
        named = any(names)
        matches = [score_attributes(attrs, d, n if named else None) for d, n in zip(self.dinos, names)]
        final = [w.alpha * float(dn) + w.beta * m.score - w.gamma * m.penalty
                 + (POPULARITY_BONUS if d.nome in POPULAR else 0.0)
                 for d, dn, m in zip(self.dinos, dense_n, matches)]
        order = sorted(range(len(self.dinos)), key=lambda i: -final[i])[:top_k]
        best = (w.alpha + (0 if attrs.is_empty() and not named else w.beta)) or 1.0
        return [
            Result(
                nome=self.dinos[i].nome, score_final=round(final[i], 4),
                similaridade=round(min(max(final[i] / best, 0.0), 1.0), 4), score_embedding=round(float(raw[i]), 4),
                score_atributos=round(matches[i].score, 4), penalidade=matches[i].penalty,
                tracos_correspondentes=matches[i].matched, tracos_nao_correspondentes=matches[i].missing,
                tracos_conflitantes=matches[i].conflicts, descricao_fisica=self.dinos[i].descricao_fisica,
            )
            for i in order
        ]


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read app/config.yaml."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_ranker(cfg: dict, model: str | None = None) -> Ranker:
    """Load data, embed it (cached on disk) and assemble a Ranker from config."""
    dinos = load_dinos()
    root = CONFIG_PATH.parents[1]
    index = DenseIndex(Embedder(model or cfg["model"]), dinos, root / cfg["cache_dir"])
    return Ranker(dinos, index, get_extractor(cfg["extractor"]), Weights(**cfg["weights"]), cfg["aggregation"])

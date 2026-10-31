"""Evaluate dense-only, attribute-only and hybrid ranking on eval/queries.json.

Usage:
    python -m eval.run_eval                 # three configs with config.yaml weights
    python -m eval.run_eval --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    python -m eval.run_eval --grid          # sweep beta/gamma for the hybrid
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.search.ranker import Ranker, Weights, build_ranker, load_config

QUERIES = Path(__file__).with_name("queries.json")


def rank_of_first_hit(ranked: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first acceptable answer, or None."""
    return next((i + 1 for i, n in enumerate(ranked) if n in expected), None)


def metrics(ranks: list[int | None]) -> dict[str, float]:
    """Top-1, Top-3 accuracy and MRR."""
    n = len(ranks)
    return {
        "top1": sum(r == 1 for r in ranks) / n,
        "top3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def evaluate(ranker: Ranker, queries: list[dict], dense: list[np.ndarray], w: Weights) -> dict[str, list[int | None]]:
    """Ranks per query type (plus 'todas')."""
    out: dict[str, list[int | None]] = defaultdict(list)
    for q, dv in zip(queries, dense):
        ranked = [r.nome for r in ranker.search(q["query"], top_k=len(ranker.dinos), weights=w, dense=dv)]
        r = rank_of_first_hit(ranked, q["esperados"])
        out[q["tipo"]].append(r)
        out["todas"].append(r)
    return out


def fmt(m: dict[str, float]) -> str:
    """Metrics as a markdown table cell row."""
    return f"{m['top1']:.3f} | {m['top3']:.3f} | {m['mrr']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override config model")
    ap.add_argument("--grid", action="store_true", help="sweep hybrid beta/gamma (alpha fixed at 1)")
    ap.add_argument("--failures", action="store_true", help="list hybrid misses")
    args = ap.parse_args()

    cfg = load_config()
    ranker = build_ranker(cfg, args.model)
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    dense = [ranker.dense_scores(q["query"]) for q in queries]  # embed once, reuse across configs
    hw = ranker.weights

    if args.grid:
        rows = []
        for b, g in itertools.product([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0], [0, 0.25, 0.5, 1.0, 2.0]):
            m = metrics(evaluate(ranker, queries, dense, Weights(1.0, b, g))["todas"])
            rows.append((m["mrr"], m["top1"], m["top3"], b, g))
        print("| β | γ | Top-1 | Top-3 | MRR |\n|---|---|---|---|---|")
        for mrr, t1, t3, b, g in sorted(rows, reverse=True)[:12]:
            print(f"| {b} | {g} | {t1:.3f} | {t3:.3f} | {mrr:.3f} |")
        return

    configs = {
        "só embeddings": Weights(1.0, 0.0, 0.0),
        "só atributos": Weights(0.0, hw.beta, hw.gamma),
        f"híbrido (α={hw.alpha}, β={hw.beta}, γ={hw.gamma})": hw,
    }
    print(f"modelo: {args.model or cfg['model']}  |  {len(queries)} consultas\n")
    print("| configuração | Top-1 | Top-3 | MRR |\n|---|---|---|---|")
    results = {}
    for name, w in configs.items():
        results[name] = evaluate(ranker, queries, dense, w)
        print(f"| {name} | {fmt(metrics(results[name]['todas']))} |")

    print("\nMRR por tipo de consulta:\n")
    types = [t for t in results[next(iter(configs))] if t != "todas"]
    print("| configuração | " + " | ".join(f"{t} (n={len(results[next(iter(configs))][t])})" for t in types) + " |")
    print("|---" * (len(types) + 1) + "|")
    for name, res in results.items():
        print(f"| {name} | " + " | ".join(f"{metrics(res[t])['mrr']:.3f}" for t in types) + " |")

    if args.failures:
        print("\nErros do híbrido (top-1 errado):")
        for q, dv in zip(queries, dense):
            top = ranker.search(q["query"], top_k=3, dense=dv)
            if top[0].nome not in q["esperados"]:
                print(f"  [{q['tipo']}] {q['query']!r} -> {[r.nome for r in top]} (esperado {q['esperados']}) attrs={ranker.extractor(q['query'])}")


if __name__ == "__main__":
    main()

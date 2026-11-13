import pytest

from app.search.dinos import load_dinos
from app.search.extractor import QueryAttrs, extract_rules
from app.search.ranker import Ranker, Weights, score_attributes

DINOS = {d.nome: d for d in load_dinos()}


def test_full_match_scores_one() -> None:
    q = QueryAttrs(postura="quadrúpede", dieta="herbívoro", tracos={"chifres", "escudo"})
    m = score_attributes(q, DINOS["Triceratops"])
    assert m.score == 1.0 and m.penalty == 0.0
    assert set(m.matched) == {"postura: quadrúpede", "dieta: herbívoro", "chifres", "escudo"}


def test_posture_conflict_and_missing_trait() -> None:
    m = score_attributes(QueryAttrs(postura="bípede", tracos={"chifres"}), DINOS["Triceratops"])
    assert m.score == 0.5
    assert m.conflicts == ["postura: bípede"]
    m = score_attributes(QueryAttrs(tracos={"chifres", "vela_dorsal"}), DINOS["Triceratops"])
    assert m.missing == ["vela_dorsal"] and m.score == 0.5


def test_size_is_ordinal_and_uncertain_traits_count_less() -> None:
    assert score_attributes(QueryAttrs(tamanho="gigante"), DINOS["Tyrannosaurus"]).score == pytest.approx(2 / 3)
    m = score_attributes(QueryAttrs(tracos={"penas"}), DINOS["Utahraptor"])  # penas incerto
    assert m.score == 0.75 and m.matched == ["penas (parcial)"]


def test_negated_trait_present_is_penalised_less_when_uncertain() -> None:
    q = QueryAttrs(negados={"penas"})
    assert score_attributes(q, DINOS["Velociraptor"]).penalty == 1.0
    assert score_attributes(q, DINOS["Utahraptor"]).penalty == 0.5
    m = score_attributes(q, DINOS["Tyrannosaurus"])
    assert m.penalty == 0.0 and m.matched == ["sem penas"]


def test_excluded_categories_are_penalised_and_explained() -> None:
    q = extract_rules("não é carnívoro nem bípede nem grande")
    tyrannosaurus = score_attributes(q, DINOS["Tyrannosaurus"])
    triceratops = score_attributes(q, DINOS["Triceratops"])
    assert tyrannosaurus.penalty == 3.0
    assert set(tyrannosaurus.conflicts) == {
        "sem dieta: carnívoro", "sem postura: bípede", "sem tamanho: grande",
    }
    assert triceratops.penalty == 1.0
    assert set(triceratops.matched) == {"sem dieta: carnívoro", "sem postura: bípede"}


def test_attribute_only_ranking_handles_negation() -> None:
    r = Ranker(list(DINOS.values()), index=None, extractor=extract_rules, weights=Weights(0, 1, 1))
    top = [x.nome for x in r.search("bico de pato sem crista", top_k=3)]
    assert top[0] == "Edmontosaurus"
    assert "Parasaurolophus" not in top


def test_dino_names_in_query_count_as_attribute() -> None:
    r = Ranker(list(DINOS.values()), index=None, extractor=extract_rules, weights=Weights(0, 1, 1))
    assert r.search("spinosaurus mirabilis", top_k=1)[0].nome == "Spinosaurus mirabilis"
    assert r.search("Tyrannosaurus", top_k=1)[0].nome == "Tyrannosaurus"
    for q in ("t rex", "T-Rex", "trex", "rex", "tiranossauro"):
        assert r.search(q, top_k=1)[0].nome == "Tyrannosaurus"
    assert r.search("argentina", top_k=1)[0].nome == "Argentinosaurus"
    top = r.search("spinosaurus", top_k=2)
    assert {x.nome for x in top} == {"Spinosaurus aegyptiacus", "Spinosaurus mirabilis"}
    assert top[0].score_atributos == top[1].score_atributos  # tie: dense breaks it in the hybrid
    assert "nome: Spinosaurus mirabilis (parcial)" in top[0].tracos_correspondentes + top[1].tracos_correspondentes


def test_popular_dinos_win_ties() -> None:
    r = Ranker(list(DINOS.values()), index=None, extractor=extract_rules, weights=Weights(0, 1, 1))
    assert r.search("spinosaurus", top_k=1)[0].nome == "Spinosaurus aegyptiacus"
    assert r.search("quatro patas", top_k=1)[0].nome in {"Triceratops", "Brachiosaurus", "Stegosaurus", "Ankylosaurus"}

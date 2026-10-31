from app.search.extractor import QueryAttrs, extract_rules


def test_synonyms_map_to_canonical_trait() -> None:
    assert "chifres" in extract_rules("tem um corno na cara").tracos
    assert "chifres" in extract_rules("pontas na testa").tracos
    assert "clava_cauda" in extract_rules("martelo no rabo").tracos


def test_longer_phrase_wins_over_substring() -> None:
    q = extract_rules("bico de pato e espinhos na cauda")
    assert q.tracos == {"bico_de_pato", "espinhos_cauda"}


def test_size_posture_diet() -> None:
    q = extract_rules("Quadrúpede ENORME que come plantas")
    assert (q.postura, q.tamanho, q.dieta) == ("quadrúpede", "gigante", "herbívoro")
    assert extract_rules("anda de duas ou quatro patas").postura == "facultativo"
    assert extract_rules("predador pequeno de duas patas").dieta == "carnívoro"


def test_negation_targets_first_trait_only() -> None:
    q = extract_rules("herbívoro com chifre no nariz mas sem espinhos no escudo")
    assert q.negados == {"espinhos"}
    assert q.tracos == {"chifres", "escudo"}


def test_negation_variants() -> None:
    q = extract_rules("carnívoro grande sem penas e sem vela")
    assert q.negados == {"penas", "vela_dorsal"} and not q.tracos
    assert extract_rules("não tem espinhos nos ombros").negados == {"espinhos"}
    assert extract_rules("nenhuma pena").negados == {"penas"}
    assert extract_rules("sem chifres nem escudo").negados == {"chifres", "escudo"}


def test_negation_scope_is_bounded() -> None:
    # "sem" is too far from "chifres" to negate it.
    q = extract_rules("sem dúvida um bicho muito grande e com chifres")
    assert q.tracos == {"chifres"} and not q.negados


def test_from_dict_roundtrip_for_llm_extractors() -> None:
    q = QueryAttrs.from_dict({"postura": "bípede", "tracos": ["penas", "xyz"], "negados": ["asas"]})
    assert q == QueryAttrs(postura="bípede", tracos={"penas"}, negados={"asas"})


def test_size_adjective_attached_to_trait_is_ignored() -> None:
    assert extract_rules("animal com unhas de tesoura gigantes").tamanho is None
    assert extract_rules("uma ema enorme que corre").tamanho is None
    assert extract_rules("um tatu gigante com casco").tamanho == "gigante"



def test_bare_nao_and_commas_do_not_flip_meaning() -> None:
    q = extract_rules("não sei, tem chifres")
    assert q.tracos == {"chifres"} and not q.negados
    q = extract_rules("não é pequeno, tem chifres")
    assert q.tamanho is None and q.tracos == {"chifres"}
    assert extract_rules("não é carnívoro").dieta is None


def test_size_inside_trait_phrase_and_plurals() -> None:
    assert extract_rules("herbívoro com garras gigantes").tamanho is None
    assert extract_rules("bichos grandes").tamanho == "grande"


def test_from_dict_tolerates_nulls_and_non_dicts() -> None:
    assert QueryAttrs.from_dict({"tracos": None, "negados": None}) == QueryAttrs()
    assert QueryAttrs.from_dict(["not", "a", "dict"]) == QueryAttrs()  # type: ignore[arg-type]

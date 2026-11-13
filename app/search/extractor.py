"""Structured attribute extraction from free-text Portuguese queries.

The rule-based extractor is the default. Any callable `str -> QueryAttrs` can
replace it; an LLM extractor only needs to return JSON in the shape accepted by
`QueryAttrs.from_dict`:
    {"postura": "bípede"|null, "dieta": ...|null, "tamanho": ...|null,
     "tracos": ["penas", ...], "negados": ["vela_dorsal", ...],
     "dietas_negadas": [...], "posturas_negadas": [...], "tamanhos_negados": [...]}
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Sequence

from app.search.dinos import DIETS, POSTURES, SIZES, TRAITS


@dataclass
class QueryAttrs:
    """Attributes found in a query. The negation sets hold explicitly excluded values."""

    postura: str | None = None
    dieta: str | None = None
    tamanho: str | None = None
    tracos: set[str] = field(default_factory=set)
    negados: set[str] = field(default_factory=set)
    dietas_negadas: set[str] = field(default_factory=set)
    posturas_negadas: set[str] = field(default_factory=set)
    tamanhos_negados: set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, d: dict) -> QueryAttrs:
        """Build from untrusted JSON (e.g. LLM output), dropping out-of-vocabulary values."""
        if not isinstance(d, dict):
            return cls()

        def pick(key: str, vocab: tuple[str, ...]) -> str | None:
            return d.get(key) if d.get(key) in vocab else None
        return cls(
            postura=pick("postura", POSTURES), dieta=pick("dieta", DIETS), tamanho=pick("tamanho", SIZES),
            tracos={t for t in d.get("tracos") or [] if t in TRAITS},
            negados={t for t in d.get("negados") or [] if t in TRAITS},
            dietas_negadas={v for v in d.get("dietas_negadas") or [] if v in DIETS},
            posturas_negadas={v for v in d.get("posturas_negadas") or [] if v in POSTURES},
            tamanhos_negados={v for v in d.get("tamanhos_negados") or [] if v in SIZES},
        )

    def is_empty(self) -> bool:
        """True when nothing structured was recognised."""
        return not (
            self.postura or self.dieta or self.tamanho or self.tracos or self.negados
            or self.dietas_negadas or self.posturas_negadas or self.tamanhos_negados
        )


Extractor = Callable[[str], QueryAttrs]

# Patterns run on lowercase, accent-free text. Order matters only within the
# single-valued fields (first match wins); traits use longest-match.
TRAIT_LEXICON: dict[str, list[str]] = {
    "chifres": [r"chifres?", r"chifrud[oa]", r"cornos?", r"pontas? na (cara|testa)"],
    "escudo": [r"escudo", r"gola", r"babado", r"folho", r"colar osseo"],
    "crista": [r"cristas?", r"tubo (na|saindo da) cabeca", r"trompete", r"trombone", r"cimitarra",
                r"machad(o|inha) na cabeca"],
    "placas": [r"placas?"],
    "espinhos": [r"espinhos?", r"espetos?", r"espetad[oa]", r"espinhent[oa]"],
    "espinhos_cauda": [r"(espinhos?|espetos?|pontas?) n[ao] (ponta d[ao] )?(cauda|rabo)", r"(cauda|rabo) (pontud[oa]|espinhos[oa])"],
    "vela_dorsal": [r"velas?", r"barbatana", r"leque nas costas", r"corcova"],
    "armadura": [r"armadura", r"blindad[oa]", r"couraca", r"casco", r"carapaca", r"osteodermos?"],
    "clava_cauda": [r"clava", r"marreta", r"(martelo|porrete|bola de osso) n[ao] (cauda|rabo)"],
    "penas": [r"penas?", r"plumagem", r"emplumad[oa]", r"penugem", r"peludo"],
    "asas": [r"asas?", r"voa", r"voava", r"planava"],
    "pescoco_longo": [r"pescoco (longo|comprido)", r"pescocud[oa]", r"girafa", r"sauropode"],
    "cauda_chicote": [r"((cauda|rabo) (de|em|como) )?chicote"],
    "patas_dianteiras_longas": [r"(patas|pernas|membros) (da frente|dianteir[oa]s) (mais )?(longas|longos|maiores|compridas)"],
    "garras_grandes": [r"garras?( (grandes?|enormes?|gigantes?|de foice))?", r"unhas?( (curvas?|em gancho|de tesoura|enormes?))?"],
    "espinho_polegar": [r"(espinho|espeto|prego) n[ao] (polegar|dedao|mao)", r"polegar (pontudo|em forma de espeto)"],
    "cabeca_domo": [r"domo", r"cabeca dura", r"careca", r"cabecadas?", r"cabeca redonda"],
    "bico": [r"bicos?", r"(bico|boca) de papagaio"],
    "bico_de_pato": [r"(bico|boca) de pato", r"(parece|como) um pato", r"bico achatado", r"hadrossauro"],
    "bracos_curtos": [r"bracos? (curtos?|pequenos?|atrofiados?)", r"bracinhos?", r"maozinhas?", r"membros anteriores (curtos|atrofiados)"],
    "cabeca_grande": [r"cabecao", r"cabecud[oa]", r"(cabeca|cranio) (grande|enorme)"],
    "focinho_longo": [r"(focinho|cara|mandibula|boca) de (crocodilo|jacare)", r"focinho (longo|comprido|estreito)"],
    "pernas_longas": [r"pernas (longas|compridas)", r"corre muito", r"corredor", r"velo[zc]", r"avestruz", r"\bema\b"],
    "semiaquatico": [r"(semi)?aquatico", r"nadava", r"nadador", r"vive na agua"],
}
SIZE_LEXICON: list[tuple[str, str]] = [
    # Explicit comparisons first: "raptor enorme do tamanho de um urso" -> médio.
    ("pequeno", r"tamanho de (um|uma) (galinha|gato|peru|cachorro|corvo|ovelha|pombo)|cabe no colo"),
    ("médio", r"tamanho de (um|uma) (homem|pessoa|cavalo|urso)"),
    ("gigante", r"tamanho de (um|uma|varios) (onibus|predio|trem)"),
    ("gigante", r"enorme|gigantesc[oa]|gigantes?|colossal|imens[oa]|o maior|mais pesado"),
    ("pequeno", r"pequen[oa]s?|pequenininh[oa]s?|minuscul[oa]s?|nanic[oa]s?"),
    ("médio", r"medios?|medianos?"),
    ("grande", r"grandes?|grandao|grandalhao"),
]
POSTURE_LEXICON: list[tuple[str, str]] = [
    ("facultativo", r"(duas|2) ou (quatro|4) (patas|pernas)|de quatro ou de duas"),
    ("bípede", r"bipede|(duas|2) (patas|pernas)|anda em pe|teropode"),
    ("quadrúpede", r"quadrupede|(quatro|4) patas|sauropode"),
]
DIET_LEXICON: list[tuple[str, str]] = [
    ("piscívoro", r"piscivoro|com(e|ia) peixes?|pescava|pescador"),
    ("onívoro", r"onivoro|come de tudo"),
    ("herbívoro", r"herbivor[oa]|vegetarian[oa]|com(e|ia|em) (plantas|folhas|vegetais)"),
    ("carnívoro", r"carnivor[oa]|predador|cacador|com(e|ia) carne|raptor|teropode"),
]

# Bare "não" only negates a directly following categorical phrase.
NEGATION_CUE = re.compile(r"\b(sem|nao tem|nao possui|nao ha|nenhum|nenhuma)\b")
SCOPE_BREAK = re.compile(r"\b(com|mas|porem)\b|[.;,]")
NEGATED_PREFIX = re.compile(r"\bnao(?:\s+(?:(?:e|eh|era)(?:\s+(?:um|uma))?|anda(?:\s+de)?))?\s*$")
CHAIN = re.compile(r"^[\s,]*(nem|ou|e)?[\s,]*$")
MAX_NEG_GAP = 3  # words allowed between the cue and the negated trait

_TRAIT_PATTERNS = [(t, re.compile(rf"\b(?:{p})\b")) for t, ps in TRAIT_LEXICON.items() for p in ps]


def normalize(text: str) -> str:
    """Lowercase and strip accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


Span = tuple[int, int, str]


def _category_spans(lexicon: list[tuple[str, str]], text: str, traits: Sequence[Span] = ()) -> list[Span]:
    """Find categorical phrases in lexicon priority order, ignoring size adjectives on traits."""
    trait_ends = {end for _, end, _ in traits}
    spans: list[Span] = []
    for label, pattern in lexicon:
        for match in re.finditer(rf"\b(?:{pattern})\b", text):
            start, end = match.span()
            if any(s <= start < e for s, e, _ in traits) or len(text[:start].rstrip()) in trait_ends:
                continue
            if not any(start < e and s < end for s, e, _ in spans):
                spans.append((start, end, label))
    return spans


def _trait_spans(text: str) -> list[Span]:
    """Non-overlapping trait matches, preferring the longest at each position."""
    found = [(m.start(), m.end(), t) for t, rx in _TRAIT_PATTERNS for m in rx.finditer(text)]
    found.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    spans: list[Span] = []
    for s in found:
        if not spans or s[0] >= spans[-1][1]:
            spans.append(s)
    return spans


def _negated(text: str, traits: list[Span], categories: list[Span]) -> set[Span]:
    """Find excluded phrases across traits and categories, including coordinated lists."""
    spans = sorted(set(traits + categories))
    groups: list[list[Span]] = []
    for span in spans:
        if groups and span[0] < max(end for _, end, _ in groups[-1]):
            groups[-1].append(span)
        else:
            groups.append([span])

    bounds = [(group[0][0], max(end for _, end, _ in group)) for group in groups]
    negated: set[Span] = set()
    previous_negated = False
    for i, (start, _) in enumerate(bounds):
        previous_end = bounds[i - 1][1] if i else 0
        before = text[previous_end:start]
        cues = list(NEGATION_CUE.finditer(before))
        if previous_negated and CHAIN.fullmatch(before):
            negated.update(groups[i])
        elif cues:
            gap = before[cues[-1].end():]
            if len(re.findall(r"\w+", gap)) <= MAX_NEG_GAP and not SCOPE_BREAK.search(gap):
                negated.update(groups[i])
        elif NEGATED_PREFIX.search(before):
            negated.update(span for span in groups[i] if span in categories)
        previous_negated = any(span in negated for span in groups[i])
    return negated


def extract_rules(query: str) -> QueryAttrs:
    """Rule/lexicon-based extractor for Portuguese queries."""
    text = normalize(query)
    traits = _trait_spans(text)
    postures = _category_spans(POSTURE_LEXICON, text)
    diets = _category_spans(DIET_LEXICON, text)
    sizes = _category_spans(SIZE_LEXICON, text, traits)
    negated = _negated(text, traits, postures + diets + sizes)

    def split(spans: list[Span]) -> tuple[list[str], set[str]]:
        excluded = {span[2] for span in spans if span in negated}
        return [label for _, _, label in spans if label not in excluded], excluded

    posturas, posturas_negadas = split(postures)
    dietas, dietas_negadas = split(diets)
    tamanhos, tamanhos_negados = split(sizes)
    tracos, negados = split(traits)
    return QueryAttrs(
        postura=next(iter(posturas), None), dieta=next(iter(dietas), None),
        tamanho=next(iter(tamanhos), None), tracos=set(tracos), negados=negados,
        posturas_negadas=posturas_negadas, dietas_negadas=dietas_negadas,
        tamanhos_negados=tamanhos_negados,
    )


def get_extractor(name: str) -> Extractor:
    """Resolve the extractor named in config. Plug an LLM-backed one in here."""
    if name == "rules":
        return extract_rules
    raise NotImplementedError(f"extractor {name!r}: implement a callable returning QueryAttrs.from_dict(llm_json)")

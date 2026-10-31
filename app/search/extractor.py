"""Structured attribute extraction from free-text Portuguese queries.

The rule-based extractor is the default. Any callable `str -> QueryAttrs` can
replace it; an LLM extractor only needs to return JSON in the shape accepted by
`QueryAttrs.from_dict`:
    {"postura": "bípede"|null, "dieta": ...|null, "tamanho": ...|null,
     "tracos": ["penas", ...], "negados": ["vela_dorsal", ...]}
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Sequence

from app.search.dinos import DIETS, POSTURES, SIZES, TRAITS


@dataclass
class QueryAttrs:
    """Attributes found in a query. `negados` are traits the user explicitly excluded."""

    postura: str | None = None
    dieta: str | None = None
    tamanho: str | None = None
    tracos: set[str] = field(default_factory=set)
    negados: set[str] = field(default_factory=set)

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
        )

    def is_empty(self) -> bool:
        """True when nothing structured was recognised."""
        return not (self.postura or self.dieta or self.tamanho or self.tracos or self.negados)


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

# Bare "não" is not a trait cue ("não sei, tem chifres"); it only blocks a
# directly following diet/posture/size ("não é carnívoro" -> unspecified).
NEGATION_CUE = re.compile(r"\b(sem|nao tem|nao possui|nao ha|nenhum|nenhuma)\b")
SCOPE_BREAK = re.compile(r"\b(com|mas|porem)\b|[.;,]")
NEGATED_PREFIX = re.compile(r"\bnao( (e|eh|era|tem|anda))?\s*$")
CHAIN = re.compile(r"^[\s,]*(nem|ou|e)?[\s,]*$")
MAX_NEG_GAP = 3  # words allowed between the cue and the negated trait

_TRAIT_PATTERNS = [(t, re.compile(rf"\b(?:{p})\b")) for t, ps in TRAIT_LEXICON.items() for p in ps]


def normalize(text: str) -> str:
    """Lowercase and strip accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def _first(lexicon: list[tuple[str, str]], text: str, spans: Sequence[tuple[int, int, str]] = ()) -> str | None:
    """First label whose pattern matches outside a negation ("não é ...") and outside
    or right after a trait phrase ("garras gigantes" describes the claws, not the animal)."""
    trait_ends = {e for _, e, _ in spans}
    for label, p in lexicon:
        for m in re.finditer(rf"\b(?:{p})\b", text):
            before = text[:m.start()]
            in_trait = any(s <= m.start() < e for s, e, _ in spans) or len(before.rstrip()) in trait_ends
            if not in_trait and not NEGATED_PREFIX.search(before):
                return label
    return None


def _trait_spans(text: str) -> list[tuple[int, int, str]]:
    """Non-overlapping trait matches, preferring the longest at each position."""
    found = [(m.start(), m.end(), t) for t, rx in _TRAIT_PATTERNS for m in rx.finditer(text)]
    found.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    spans: list[tuple[int, int, str]] = []
    for s in found:
        if not spans or s[0] >= spans[-1][1]:
            spans.append(s)
    return spans


def _negated(text: str, spans: list[tuple[int, int, str]]) -> set[int]:
    """Indices of spans in a negation scope: the first trait after a cue, plus any chained by e/nem/ou."""
    neg: set[int] = set()
    for cue in NEGATION_CUE.finditer(text):
        i = next((k for k, s in enumerate(spans) if s[0] >= cue.end()), None)
        if i is None:
            continue
        between = text[cue.end():spans[i][0]]
        if len(re.findall(r"\w+", between)) > MAX_NEG_GAP or SCOPE_BREAK.search(between):
            continue
        neg.add(i)
        while i + 1 < len(spans) and CHAIN.match(text[spans[i][1]:spans[i + 1][0]]):
            i += 1
            neg.add(i)
    return neg


def extract_rules(query: str) -> QueryAttrs:
    """Rule/lexicon-based extractor for Portuguese queries."""
    text = normalize(query)
    spans = _trait_spans(text)
    neg = _negated(text, spans)
    negados = {spans[i][2] for i in neg}
    tracos = {s[2] for k, s in enumerate(spans) if k not in neg} - negados
    return QueryAttrs(
        postura=_first(POSTURE_LEXICON, text), dieta=_first(DIET_LEXICON, text),
        tamanho=_first(SIZE_LEXICON, text, spans), tracos=tracos, negados=negados,
    )


def get_extractor(name: str) -> Extractor:
    """Resolve the extractor named in config. Plug an LLM-backed one in here."""
    if name == "rules":
        return extract_rules
    raise NotImplementedError(f"extractor {name!r}: implement a callable returning QueryAttrs.from_dict(llm_json)")

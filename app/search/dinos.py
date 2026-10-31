"""Dino dataset: closed vocabularies, loading and schema validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "dinos.json"

DIETS = ("carnívoro", "herbívoro", "onívoro", "piscívoro")
SIZES = ("pequeno", "médio", "grande", "gigante")  # ordinal: index = rank
POSTURES = ("bípede", "quadrúpede", "facultativo")
TRAITS = (
    "chifres", "escudo", "crista", "placas", "espinhos", "espinhos_cauda",
    "vela_dorsal", "armadura", "clava_cauda", "penas", "asas", "pescoco_longo",
    "cauda_chicote", "patas_dianteiras_longas", "garras_grandes",
    "espinho_polegar", "cabeca_domo", "bico", "bico_de_pato", "bracos_curtos",
    "cabeca_grande", "focinho_longo", "pernas_longas", "semiaquatico",
)
REQUIRED = ("nome", "periodo", "dieta", "tamanho", "postura", "tracos", "descricao_fisica", "parafrases")
# Size category bounds in metres (upper bound exclusive).
SIZE_BOUNDS = {"pequeno": (0, 2.5), "médio": (2.5, 7.5), "grande": (7.5, 15), "gigante": (15, 1e9)}


@dataclass(frozen=True)
class Dino:
    """One dinosaur record."""

    nome: str
    periodo: str
    dieta: str
    comprimento_m: float
    tamanho: str
    postura: str
    tracos: dict[str, bool]  # trait -> incerto
    descricao_fisica: str
    parafrases: list[str]
    notas: str = ""
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def passages(self) -> list[str]:
        """Texts indexed for this dino: main description first, then paraphrases."""
        return [self.descricao_fisica, *self.parafrases]


def validate(raw: list[dict]) -> list[str]:
    """Return a list of schema errors (empty when valid)."""
    errs: list[str] = []
    names = [d.get("nome") for d in raw]
    if len(raw) != 30:
        errs.append(f"expected 30 dinos, got {len(raw)}")
    if len(set(names)) != len(names):
        errs.append("duplicate names")
    for d in raw:
        n = d.get("nome", "?")
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            errs.append(f"{n}: missing {missing}")
            continue
        if d["dieta"] not in DIETS:
            errs.append(f"{n}: bad dieta {d['dieta']!r}")
        if d["postura"] not in POSTURES:
            errs.append(f"{n}: bad postura {d['postura']!r}")
        cat, length = d["tamanho"].get("categoria"), d["tamanho"].get("comprimento_m")
        if cat not in SIZES:
            errs.append(f"{n}: bad categoria {cat!r}")
        elif not isinstance(length, (int, float)) or not SIZE_BOUNDS[cat][0] <= length < SIZE_BOUNDS[cat][1]:
            errs.append(f"{n}: comprimento {length} outside {cat} bounds {SIZE_BOUNDS[cat]}")
        for t in d["tracos"]:
            if t.get("traco") not in TRAITS:
                errs.append(f"{n}: unknown trait {t.get('traco')!r}")
            if not isinstance(t.get("incerto"), bool):
                errs.append(f"{n}: trait {t.get('traco')!r} missing boolean incerto")
        sentences = [s for s in d["descricao_fisica"].split(". ") if s.strip()]
        if not 3 <= len(sentences) <= 5:
            errs.append(f"{n}: descricao_fisica has {len(sentences)} sentences")
        if not 3 <= len(d["parafrases"]) <= 5:
            errs.append(f"{n}: {len(d['parafrases'])} parafrases")
    return errs


def load_dinos(path: Path = DATA_PATH) -> list[Dino]:
    """Load and validate the dataset; raise ValueError on schema errors."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if errs := validate(raw):
        raise ValueError("invalid dinos.json:\n" + "\n".join(errs))
    return [
        Dino(
            nome=d["nome"], periodo=d["periodo"], dieta=d["dieta"],
            comprimento_m=float(d["tamanho"]["comprimento_m"]), tamanho=d["tamanho"]["categoria"],
            postura=d["postura"], tracos={t["traco"]: t["incerto"] for t in d["tracos"]},
            descricao_fisica=d["descricao_fisica"], parafrases=d["parafrases"],
            notas=d.get("notas", ""), raw=d,
        )
        for d in raw
    ]


if __name__ == "__main__":
    dinos = load_dinos()
    print(f"OK: {len(dinos)} dinos, {sum(len(d.passages) for d in dinos)} passages")

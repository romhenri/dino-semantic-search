# Dino Search by Similarity

Busca semântica de dinossauros a partir de descrições físicas livres em português.
Você escreve "quadrúpede enorme com placas nas costas e espinhos na cauda" e recebe os
dinossauros mais parecidos, com score e justificativa por traço.

## Como rodar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app            # http://localhost:8000
```

Na primeira execução o modelo (~1,1 GB) é baixado e os embeddings da base são
calculados e salvos em `.cache/`. Nas seguintes, o cache é reaproveitado e só é
recalculado se `data/dinos.json` ou o modelo mudarem (a chave é um hash dos dois).

```bash
pip install -r requirements-dev.txt
python -m app.search.dinos      # valida data/dinos.json
pytest                          # testes (o de API carrega o modelo)
mypy app eval tests --ignore-missing-imports
python -m eval.run_eval         # tabela das três configurações
python -m eval.run_eval --grid  # varredura de β/γ
python -m eval.run_eval --failures
python -m eval.run_eval --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

### API

- `POST /search` com `{"query": str, "top_k": int = 5}` retorna uma lista com `nome`, `score_final`,
  `similaridade` (score_final em [0, 1], para a barra), `score_embedding` (cosseno bruto),
  `score_atributos`, `penalidade`, `tracos_correspondentes`, `tracos_nao_correspondentes`,
  `tracos_conflitantes` e `descricao_fisica`.
- `GET /dinos` lista a base.
- `GET /` serve o front-end (`static/index.html`).

## Estrutura

```
data/dinos.json            30 dinossauros, vocabulário fechado de 24 traços
app/search/dinos.py        carregamento + validação do schema
app/search/embedder.py     embeddings (E5) + índice NumPy com cache em disco
app/search/extractor.py    extração de postura/dieta/tamanho/traços/negações por regras
app/search/ranker.py       fusão de scores + explicação por traço
app/config.yaml            modelo, agregação, pesos
app/main.py                FastAPI
static/index.html          front-end
eval/queries.json          52 consultas de teste
eval/run_eval.py           Top-1, Top-3, MRR
tests/                     pytest
```

## Decisões de design

**Dados.** Cada dinossauro tem `tracos` como lista de `{"traco", "incerto"}` dentro de um
vocabulário fechado (`TRAITS` em `dinos.py`), o que permite casar a consulta com a base sem
ambiguidade. Traços debatidos levam `"incerto": true`: penas em Deinonychus, Utahraptor,
Gallimimus, Therizinosaurus e Compsognathus (inferidas por parentesco, sem evidência direta),
Oviraptor (inferidas por parentes, como a crista, mal conhecida no holótipo esmagado) e o
hábito semiaquático do Spinosaurus. Velociraptor (marcas de inserção de penas na ulna) e
Microraptor (plumagem preservada) têm penas sem marca de incerteza. Outros debates vão
no campo `notas` (postura do Spinosaurus, validade do gênero Brontosaurus, posição do espinho
lateral do Kentrosaurus). As categorias de tamanho são validadas contra o comprimento:
pequeno < 2,5 m, médio < 7,5 m, grande < 15 m, gigante acima disso.

**Modelo.** `intfloat/multilingual-e5-base`, com os prefixos `query:` e `passage:`. Ele foi
treinado para recuperação assimétrica (consulta curta contra passagem longa), que é
exatamente o caso aqui. Na avaliação ele ganhou do `paraphrase-multilingual-mpnet-base-v2`
com folga no modo só embeddings (MRR 0,825 contra 0,609).

**Busca densa.** A descrição e cada paráfrase são vetores separados (150 no total). O score
de um dino é a média dos seus 2 melhores vetores (`top2_mean`), que superou o máximo
(MRR 0,825 contra 0,765 só com embeddings, e 0,981 contra 0,962 no híbrido): um único
vetor parecido por acaso pesa menos. O índice é uma multiplicação de matrizes em NumPy,
isolada em `DenseIndex`, onde dá para trocar por FAISS ou pgvector se a base crescer.

**Extrator.** Léxico de sinônimos em regex sobre texto sem acento e em minúsculas
(`"corno"`, `"pontas na testa"` viram `chifres`; `"martelo no rabo"` vira `clava_cauda`). Nos
traços vence o casamento mais longo (`"bico de pato"` ganha de `"bico"`). Negação:
uma pista (`sem`, `não tem`, `nenhum`...) nega o primeiro traço que aparece até 3
palavras depois dela, mais os encadeados por `e`/`nem`/`ou`, e para em `com`/`mas`/vírgula.
Assim, "sem espinhos no escudo" nega só `espinhos`. Um "não" solto não nega traços ("não sei,
tem chifres"), mas anula uma dieta, postura ou tamanho logo em seguida ("não é carnívoro"
deixa a dieta sem valor, em vez de casar com carnívoro). Adjetivos de tamanho dentro ou logo
depois de um traço não contam como tamanho do animal ("garras gigantes"). Comparações explícitas ("do tamanho
de um urso") têm prioridade sobre adjetivos. Para trocar por um LLM, basta qualquer
`Callable[[str], QueryAttrs]` registrado em `get_extractor`; `QueryAttrs.from_dict` valida o
JSON do LLM contra o vocabulário.

**Fusão.**
`score_final = α · denso_normalizado + β · match_atributos − γ · penalidade_negação`

- `denso_normalizado`: cosseno reescalado por consulta para [0, 1] (min-max entre os 30
  dinos). Os cossenos do E5 ficam numa faixa estreita (~0,75 a 0,9); sem isso, os pesos não
  seriam comparáveis.
- `match_atributos`: média sobre os atributos que a consulta menciona. Traço presente vale
  1 (0,75 se incerto). Postura e dieta valem 1 se iguais e 0,5 se compatíveis (facultativo
  contra bípede, piscívoro contra carnívoro, onívoro contra os dois); com 0, contam como
  conflito. Tamanho é ordinal: 1 − distância/3.
- `penalidade`: +1 para cada traço negado que o dino tem (+0,5 se incerto).

**Pesos escolhidos: α = 1,0, β = 1,5, γ = 1,0.** A varredura (`--grid`) mostra um platô
de MRR 0,981 para todo β ≥ 1,5 e γ ≥ 0,5. Abaixo dele a qualidade cai (β = γ = 0,25: MRR
0,922; β = 1, γ = 0,5: 0,968; γ = 0 em qualquer β: cerca de 0,92, ou seja, sem penalidade
as negações quebram). Escolhi a borda esquerda do platô: o menor peso de atributos que
atinge o máximo, para o denso continuar o mais influente possível. Na prática, os
atributos definem as faixas e o denso ordena dentro delas e decide sozinho quando o
extrator não reconhece nada (aí `match_atributos` é 0 para todos e o híbrido vira só
embeddings).

**Explicabilidade.** Cada resultado lista os traços que bateram (com `(parcial)` quando
o match é parcial, como um traço incerto ou um tamanho adjacente), os que não bateram e os
que conflitam (postura ou dieta incompatível, tamanho a 2+ categorias de distância, traço
negado presente).

## Resultados

52 consultas em `eval/queries.json`: 17 diretas, 17 leigas, 11 com negação e 7 ambíguas
(com várias respostas aceitáveis; conta o primeiro acerto). Modelo E5-base, agregação
`top2_mean`.

| configuração | Top-1 | Top-3 | MRR |
|---|---|---|---|
| só embeddings | 0,712 | 0,923 | 0,825 |
| só atributos | 0,846 | 0,962 | 0,903 |
| **híbrido** (α=1, β=1,5, γ=1) | **0,962** | **1,000** | **0,981** |

MRR por tipo:

| configuração | direta | leiga | negação | ambígua |
|---|---|---|---|---|
| só embeddings | 0,941 | 0,855 | 0,578 | 0,857 |
| só atributos | 0,902 | 0,887 | 0,867 | 1,000 |
| híbrido | 0,971 | 1,000 | 0,955 | 1,000 |

Comparação de modelos (híbrido com os mesmos pesos):

| modelo | só embeddings (MRR) | híbrido (MRR) |
|---|---|---|
| multilingual-e5-base | 0,825 | 0,981 |
| paraphrase-multilingual-mpnet-base-v2 | 0,609 | 0,942 |

O que os números mostram: embeddings puros vão bem em descrições diretas mas quebram em
negação (0,578), porque "sem penas" fica perto dos dinos com penas. Os atributos resolvem a
negação e a ambiguidade, mas sozinhos não desempatam dinos com o mesmo conjunto de traços.
O híbrido junta os dois.

Erros restantes do híbrido (top-1):
- "crista tubular longa curvada para trás e bico achatado": Corythosaurus em 1º,
  Parasaurolophus em 2º. O léxico não distingue o formato da crista e o denso não
  compensou.
- "parecido com um Triceratops mas sem chifres": Protoceratops em 2º. O denso não sabe
  usar o nome de outro dinossauro como referência.

### Testado e descartado

Remover os trechos negados da consulta antes de embuti-la ("carnívoro grande sem penas"
vira "carnívoro grande") parecia óbvio, mas não mudou nada com os pesos finais (0,981 nos
dois casos) e piorou as negações com β baixo. Foi removido.

### Limitações

- **Viés do autor.** A base, as paráfrases, o léxico e as consultas de teste foram escritos
  pela mesma pessoa (no caso, o modelo que gerou o projeto). As consultas usam vocabulário
  diferente das paráfrases, mas o léxico do extrator foi escrito conhecendo as consultas, e
  os pesos foram ajustados nesse mesmo conjunto. Os números de "só atributos" e "híbrido"
  são otimistas. Uma avaliação honesta precisa de consultas escritas por outras pessoas.
- O extrator ignora negação de postura ou dieta ("não é carnívoro" só deixa a dieta sem
  valor, não penaliza carnívoros) e só entende comparações de tamanho de uma lista fixa.
- O denso é normalizado por consulta (min-max), então o melhor resultado sempre recebe 1 no
  componente denso, mesmo para uma consulta sem sentido. A `similaridade` mostrada é relativa
  ao melhor score possível para aquela consulta, não uma probabilidade.
- `DenseIndex` é uma única classe concreta. Trocar por FAISS significa reescrever
  `scores()` (a multiplicação de matrizes e a agregação), sem mudar o resto.
- 30 itens: NumPy de força bruta é o certo aqui; a abstração `DenseIndex` existe para o
  dia em que não for.

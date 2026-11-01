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
data/dinos.json            50 dinossauros, vocabulário fechado de 24 traços
app/search/dinos.py        carregamento + validação do schema
app/search/embedder.py     embeddings (E5) + índice NumPy com cache em disco
app/search/extractor.py    extração de postura/dieta/tamanho/traços/negações por regras
app/search/ranker.py       fusão de scores + explicação por traço
app/config.yaml            modelo, agregação, pesos
app/main.py                FastAPI
static/index.html          front-end
static/img/                ilustrações (PNG, 320px) + credits.json
eval/queries.json          66 consultas de teste
eval/run_eval.py           Top-1, Top-3, MRR
tests/                     pytest
```

## Decisões de design

**Dados.** Cada dinossauro tem `tracos` como lista de `{"traco", "incerto"}` dentro de um
vocabulário fechado (`TRAITS` em `dinos.py`), o que permite casar a consulta com a base sem
ambiguidade. Traços debatidos levam `"incerto": true`: penas em Deinonychus, Utahraptor,
Gallimimus, Therizinosaurus e Compsognathus (inferidas por parentesco, sem evidência direta),
Oviraptor (inferidas por parentes, como a crista, mal conhecida no holótipo esmagado) e o
hábito semiaquático do Spinosaurus aegyptiacus. Velas inferidas também são incertas: Ouranosaurus (vela
ou corcova), Amargasaurus (velas de pele ou pontas de queratina) e Spinosaurus mirabilis, cujo
formato de vela é desconhecido. Velociraptor (marcas de inserção de penas na ulna) e
Microraptor (plumagem preservada) têm penas sem marca de incerteza. Outros debates vão
no campo `notas` (postura do Spinosaurus aegyptiacus, validade do gênero Brontosaurus, posição do espinho
lateral do Kentrosaurus, material imaturo do Spinosaurus mirabilis).

A base tem os 30 dinossauros da lista original mais 20: Spinosaurus mirabilis (descrito por
Sereno et al. em fevereiro de 2026 na *Science*), Ceratosaurus, Albertosaurus,
Majungasaurus, Coelophysis, Herrerasaurus, Suchomimus, Yutyrannus, Archaeopteryx,
Ouranosaurus, Maiasaura, Lambeosaurus, Pachyrhinosaurus, Psittacosaurus, Euoplocephalus,
Borealopelta, Apatosaurus, Camarasaurus, Amargasaurus e Mamenchisaurus. Vários foram
escolhidos por criar confusões úteis para a avaliação: Ouranosaurus é herbívoro com vela,
Euoplocephalus também tem clava, Suchomimus também tem cara de crocodilo e Pachyrhinosaurus é
um ceratopsiano sem chifres. As categorias de tamanho são validadas contra o comprimento:
pequeno < 2,5 m, médio < 7,5 m, grande < 15 m, gigante acima disso.

**Modelo.** `intfloat/multilingual-e5-base`, com os prefixos `query:` e `passage:`. Ele foi
treinado para recuperação assimétrica (consulta curta contra passagem longa), que é
exatamente o caso aqui. Na avaliação ele ganhou do `paraphrase-multilingual-mpnet-base-v2`
com folga no modo só embeddings (MRR 0,807 contra 0,616).

**Busca densa.** A descrição e cada paráfrase são vetores separados (250 no total). O score
de um dino é a média dos seus 2 melhores vetores (`top2_mean`), que superou o máximo
(MRR 0,814 contra 0,760 só com embeddings, e 0,967 contra 0,944 no híbrido): um único
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

- `denso_normalizado`: cosseno reescalado por consulta para [0, 1] (min-max entre os 50
  dinos). Os cossenos do E5 ficam numa faixa estreita (~0,75 a 0,9); sem isso, os pesos não
  seriam comparáveis.
- `match_atributos`: média sobre os atributos que a consulta menciona. Traço presente vale
  1 (0,75 se incerto). Postura e dieta valem 1 se iguais e 0,5 se compatíveis (facultativo
  contra bípede, piscívoro contra carnívoro, onívoro contra os dois); com 0, contam como
  conflito. Tamanho é ordinal: 1 − distância/3.
- `penalidade`: +1 para cada traço negado que o dino tem (+0,5 se incerto).
- Nome: se a consulta contém palavras do nome de algum dino, a fração do nome presente entra
  como mais um atributo. "spinosaurus" casa metade do nome das duas espécies (empate, que o
  denso desempata a favor do *aegyptiacus*); "spinosaurus mirabilis" casa o nome inteiro de
  uma só. O embedding sozinho não lê bem nomes científicos. Um nome citado como referência
  ("parecido com um Triceratops mas sem chifres") não atrapalha, porque a negação custa mais
  que o nome rende.

**Pesos escolhidos: α = 1,0, β = 1,5, γ = 1,0.** A varredura (`--grid`) mostra um platô
de MRR 0,967 para todo β ≥ 1,5 e γ ≥ 0,5. Abaixo dele a qualidade cai (β = γ = 0,25: MRR
0,931; β = 1, γ = 0,5: 0,960; γ = 0 em qualquer β: 0,932, ou seja, sem penalidade as
negações quebram). O platô e a escolha se mantiveram iguais ao passar de 30 para 50
dinossauros. Escolhi a borda esquerda do platô: o menor peso de atributos que
atinge o máximo, para o denso continuar o mais influente possível. Na prática, os
atributos definem as faixas e o denso ordena dentro delas e decide sozinho quando o
extrator não reconhece nada (aí `match_atributos` é 0 para todos e o híbrido vira só
embeddings).

**Explicabilidade.** Cada resultado lista os traços que bateram (com `(parcial)` quando
o match é parcial, como um traço incerto ou um tamanho adjacente), os que não bateram e os
que conflitam (postura ou dieta incompatível, tamanho a 2+ categorias de distância, traço
negado presente).

## Resultados

66 consultas em `eval/queries.json`: 22 diretas, 21 leigas, 14 com negação e 9 ambíguas
(com várias respostas aceitáveis; conta o primeiro acerto). Modelo E5-base, agregação
`top2_mean`.

| configuração | Top-1 | Top-3 | MRR |
|---|---|---|---|
| só embeddings | 0,712 | 0,894 | 0,807 |
| só atributos | 0,727 | 0,833 | 0,792 |
| **híbrido** (α=1, β=1,5, γ=1) | **0,939** | **1,000** | **0,967** |

MRR por tipo:

| configuração | direta | leiga | negação | ambígua |
|---|---|---|---|---|
| só embeddings | 0,932 | 0,823 | 0,568 | 0,837 |
| só atributos | 0,790 | 0,727 | 0,826 | 0,897 |
| híbrido | 0,977 | 0,952 | 0,952 | 1,000 |

Comparação de modelos (híbrido com os mesmos pesos):

| modelo | só embeddings (MRR) | híbrido (MRR) |
|---|---|---|
| multilingual-e5-base | 0,807 | 0,967 |
| paraphrase-multilingual-mpnet-base-v2 | 0,616 | 0,908 |

O que os números mostram: embeddings puros vão bem em descrições diretas mas quebram em
negação (0,568), porque "sem penas" fica perto dos dinos com penas. Os atributos resolvem a
negação e a ambiguidade, mas sozinhos não desempatam dinos com o mesmo conjunto de traços.
Isso piorou de 30 para 50 dinossauros (MRR de "só atributos" caiu de 0,903 para 0,792),
porque agora mais dinos compartilham os mesmos traços. O híbrido junta os dois e caiu bem
menos (0,981 para 0,967).

Ao ampliar a base, algumas consultas antigas ganharam respostas novas igualmente corretas
(por exemplo, Suchomimus em "come peixe" e Euoplocephalus em "tatu gigante com martelo no
rabo"), e "parecido com um Triceratops mas sem chifres" passou a aceitar o Pachyrhinosaurus,
que é literalmente isso.

Erros restantes do híbrido (top-1); deixados como estão para não ajustar o léxico ao teste:
- "crista tubular longa curvada para trás e bico achatado": Corythosaurus em 1º,
  Parasaurolophus em 2º. O léxico não distingue o formato da crista.
- "girafa pré-histórica": Mamenchisaurus em 1º, Brachiosaurus em 2º. "Girafa" vira só
  `pescoco_longo`, sem a postura inclinada.
- "minúsculo, cabe no colo e come lagartos": Archaeopteryx em 1º, Compsognathus em 2º.
- "ceratopsiano sem chifres com uma placa grossa no nariz": Pachyrhinosaurus em 3º. "Placa"
  vira `placas` (as do Stegosaurus), um falso positivo do léxico.

### Hub de embedding: Dilophosaurus

Em consultas genéricas de carnívoro ("carnívoro de duas patas", "predador bípede"), todos os
terópodes empatam nos atributos e o denso desempata. O Dilophosaurus ganhava quase sempre
(top-3 em 8 de 16 consultas genéricas) porque duas das suas cinco passagens eram genéricas
("predador magro com crista dupla", "Predador bípede esguio..."), e a média das 2 melhores
premiava isso. Uma correção global de hubness (subtrair a similaridade média de cada dino
contra as paráfrases dos outros) quase não ajudou (8 para 6), porque o problema era local ao
grupo dos carnívoros. A correção foi nos dados: as passagens dele agora descrevem a crista
dupla em vez de "predador de duas patas". Resultado: 0 de 16, híbrido inalterado (0,967), e
"só embeddings" caiu de 0,814 para 0,800, porque parte do acerto vinha do próprio hub. Regra
para novas entradas: paráfrases devem descrever o que distingue o animal, não a categoria.

O mesmo aconteceu entre as duas espécies de *Spinosaurus*: "espinossauro" e "spinosaurus"
traziam o *mirabilis* em 1º, porque só as passagens dele usavam o nome ("espinossauro com
crista...", "parente do Spinosaurus..."). O *aegyptiacus* passou a ter o nome popular nas
passagens ("o espinossauro clássico"), e o casamento de nome acima resolve os nomes completos.
Em 18 consultas que descrevem o espinossauro clássico, 17 trazem o *aegyptiacus* em 1º (a
exceção, "terópode de 14 metros...", falha porque o extrator não lê metros).

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
- 50 itens: NumPy de força bruta é o certo aqui; a abstração `DenseIndex` existe para o
  dia em que não for.

## Créditos das ilustrações

As ilustrações em `static/img/` vêm da [Wikimedia Commons](https://commons.wikimedia.org/),
reduzidas para 320 px de largura e convertidas para PNG. Todas têm licença Creative Commons
(CC BY ou CC BY-SA); as adaptações seguem a mesma licença do original. Autor, licença e link
de cada uma também aparecem no card e em `static/img/credits.json`.

| dinossauro | autor | licença |
|---|---|---|
| Tyrannosaurus | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Tyrannosaurus_rex_Reconstruction_by_Nobu_Tamura.jpg) |
| Triceratops | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Triceratops_BW.jpg) |
| Velociraptor | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Velociraptor_mongoliensis.jpg) |
| Stegosaurus | Nobu Tamura | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Stegosaurus_BW_transparent.png) |
| Brachiosaurus | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Brachiosaurus_NT_new.jpg) |
| Spinosaurus aegyptiacus | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Spinosaurus_life_restoration_with_Onchopristis.jpg) |
| Ankylosaurus | Emily Willoughby | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Ankylosaurus_magniventris_reconstruction.png) |
| Parasaurolophus | Connor Ashbridge | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Life_reconstruction_of_Parasaurolophus_walkeri.png) |
| Pachycephalosaurus | Fred Wierum | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Pachycephalosaurus_Reconstruction_transparent.png) |
| Allosaurus | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Allosaurus_BW.jpg) |
| Diplodocus | Nobu Tamura | [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Diplodocus_BW.jpg) |
| Iguanodon | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Iguanodon_new_NT_transparent.png) |
| Brontosaurus | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Brontosaurus_NT_small.jpg) |
| Dilophosaurus | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Dilophosaurus_NT.jpg) |
| Carnotaurus | Nobu Tamura | [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Carnotaurus_BW.jpg) |
| Giganotosaurus | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Giganotosaurus_BW.jpg) |
| Baryonyx | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Baryonyx_BW.jpg) |
| Gallimimus | PaleoNeolitic | [CC BY 4.0](https://commons.wikimedia.org/wiki/File:Gallimimus_Restoration.png) |
| Compsognathus | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Compsognathus_BW.jpg) |
| Oviraptor | PaleoNeolitic | [CC BY 4.0](https://commons.wikimedia.org/wiki/File:Oviraptor_Restoration.png) |
| Therizinosaurus | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Therizinosaurus_NT.jpg) |
| Microraptor | Fred Wierum | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Microraptor_Restoration.png) |
| Deinonychus | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Deinonychus_BW.jpg) |
| Utahraptor | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Utahraptor_BW.jpg) |
| Edmontosaurus | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Edmontosaurus_BW.jpg) |
| Styracosaurus | Nobu Tamura | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Styracosaurus_BW_transparent.png) |
| Argentinosaurus | Nobu Tamura | [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Argentinosaurus_BW.jpg) |
| Protoceratops | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Protoceratops_BW.jpg) |
| Corythosaurus | Connor Ashbridge | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Life_reconstruction_of_Corythosaurus_casuarius.png) |
| Kentrosaurus | Nobu Tamura | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Kentrosaurus_NT_transparent.png) |
| Spinosaurus mirabilis | Connor Ashbridge | [CC BY 4.0](https://commons.wikimedia.org/wiki/File:Spinosaurus_mirabilis.png) |
| Ceratosaurus | Nobu Tamura | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Ceratosaurus_NT.jpg) |
| Albertosaurus | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Albertosaurus_NT_small.jpg) |
| Majungasaurus | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Majungasaurus_BW.jpg) |
| Coelophysis | TotalDino | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Coelophysis_TD.png) |
| Herrerasaurus | Nobu Tamura | [CC BY 2.5](https://commons.wikimedia.org/wiki/File:Herrerasaurus_BW.jpg) |
| Suchomimus | Nobu Tamura | [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Suchomimus_BW.jpg) |
| Yutyrannus | Tomopteryx | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Yutyrannus_huali.png) |
| Archaeopteryx | Nobu Tamura | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Archaeopteryx_NT.jpg) |
| Ouranosaurus | Audrey.m.horn | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Ouranosaurus_nigeriensis_restoration.png) |
| Maiasaura | Nobu Tamura | [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Maiasaura_BW.jpg) |
| Lambeosaurus | Anonymous Dinonerd | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Life_restoration_of_Lambeosaurus_lambei.jpg) |
| Pachyrhinosaurus | Nobu Tamura | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Pachyrhinosaurus_BW_flipped_transparent.png) |
| Psittacosaurus | Nobu Tamura | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Psittacosaurus_sibiricus_whole_BW_transparent.png) |
| Euoplocephalus | Nobu Tamura | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Euoplocephalus_BW_transparent.png) |
| Borealopelta | Connor Ashbridge | [CC BY 4.0](https://commons.wikimedia.org/wiki/File:Borealopelta_markmitchelli.png) |
| Apatosaurus | Durbed | [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Apatosaurus_louisae_by_durbed.jpg) |
| Camarasaurus | Nobu Tamura | [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Camarasaurus_BW.jpg) |
| Amargasaurus | Nobu Tamura | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Amargasaurus_NT_small.jpg) |
| Mamenchisaurus | Connor Ashbridge | [CC BY 4.0](https://commons.wikimedia.org/wiki/File:Mamenchisaurus_sanjiangensis.png) |

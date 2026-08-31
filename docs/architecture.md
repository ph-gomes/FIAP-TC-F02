# Arquitetura

## Visão geral

O projeto otimiza os hiperparâmetros de um `RandomForestClassifier` (diagnóstico de
câncer de mama) com um Algoritmo Genético e usa o Claude (Anthropic) para traduzir
os resultados numéricos em explicações, relatórios e avaliação de qualidade em
linguagem natural.

```
Wisconsin Breast Cancer Dataset (Kaggle)
   │
   ▼
Preparação dos dados (treino/validação/teste)
   │
   ▼
Random Forest original (baseline)
   │
   ▼
Algoritmo Genético (busca de hiperparâmetros do Random Forest, usa treino/validação)
   │
   ▼
Random Forest final treinado com o melhor indivíduo
   │
   ├──▶ Avaliação no conjunto de teste (acurácia, recall, F1-score)
   │
   ▼
Camada de interpretação com Claude (explicações por caso + relatório executivo)
   │
   ▼
Avaliação de qualidade das explicações (checagens heurísticas + LLM-as-judge)
```

## Organização do código

```
.
├── src/
│   ├── ga.py             # representação, fitness, seleção, cruzamento, mutação, laço do GA
│   ├── llm.py             # prompts, payload do caso, chamadas ao Claude (explicação, relatório, juiz)
│   ├── quality.py         # checagens heurísticas sobre as explicações geradas
│   └── logging_utils.py   # logger estruturado (JSON) + decorator de monitoramento (monitor_call)
├── tests/                 # testes automatizados (pytest) das funções em src/
├── docs/
│   └── architecture.md    # este documento
└── tech_challenge_f02.ipynb   # notebook de demonstração: carrega os dados, chama src/ e mostra os resultados
```

O notebook é a camada de orquestração e demonstração: carrega os dados, chama as
funções de `src/` na ordem do pipeline acima e exibe os resultados (tabelas,
gráficos, explicações). A lógica reutilizável e testável vive em `src/`, para que
possa ser validada por `tests/` sem precisar executar o notebook inteiro (nem
consumir a API do Claude).

## Decisões de implementação

- **`calcular_fitness`/`selecionar`/`algoritmo_genetico` recebem os dados de
  treino/validação por parâmetro** (em vez de acessar `X_train`/`y_train`
  globais), para que possam ser testados com qualquer dataset, inclusive um
  sintético nos testes.
- **As funções de LLM recebem o `client` da Anthropic por parâmetro**
  (`explain_diagnosis_with_claude(client, ...)` etc.), em vez de depender de uma
  variável global `claude_client`. Isso mantém `src/llm.py` importável e
  testável mesmo sem uma `ANTHROPIC_API_KEY` configurada (só as chamadas reais à
  API exigem a chave).
- **Monitoramento e logging** (`src/logging_utils.py`): todas as chamadas à API
  do Claude passam pelo decorator `monitor_call`, que registra latência,
  sucesso/erro e grava tanto em log estruturado (JSON, `pipeline_metrics.log`)
  quanto em uma lista em memória (`performance_metrics`) — a base para
  observabilidade caso o pipeline seja escalado horizontalmente.
- **Retentativa com backoff exponencial** nas chamadas ao Claude
  (`explain_diagnosis_with_claude`, `generate_executive_summary_with_claude`)
  para tolerar rate limiting e indisponibilidades transitórias sob picos de
  demanda.

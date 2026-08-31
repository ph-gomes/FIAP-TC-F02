# FIAP-TC-F02 — Tech Challenge Fase 2

Tech Challenge da Fase 2 da Pós Tech (IA para Devs) — **Projeto 1: Otimização de Modelos de Diagnóstico**.

O hospital universitário precisa melhorar a precisão e eficiência dos modelos de diagnóstico
desenvolvidos no Módulo 1. A solução usa um **algoritmo genético** para otimizar os
hiperparâmetros de um `RandomForestClassifier` e uma **LLM (Claude)** para gerar
interpretações em linguagem natural dos resultados para profissionais de saúde.

## Estrutura do projeto

```
.
├── src/                        # lógica reutilizável e testável
│   ├── ga.py                   # algoritmo genético (representação, fitness, seleção, cruzamento, mutação)
│   ├── llm.py                  # prompts e chamadas ao Claude (explicação, relatório executivo, juiz)
│   ├── quality.py              # checagens heurísticas sobre as explicações geradas
│   └── logging_utils.py        # logger estruturado (JSON) + decorator de monitoramento
├── tests/                      # testes automatizados (pytest) das funções em src/
├── docs/
│   └── architecture.md         # documentação de arquitetura e decisões de implementação
└── tech_challenge_f02.ipynb    # notebook de demonstração: orquestra src/ e mostra os resultados
```

O notebook carrega os dados, chama as funções de `src/` na ordem do pipeline e
exibe os resultados (tabelas, gráficos, explicações). Detalhes de arquitetura e
decisões de implementação estão em [`docs/architecture.md`](docs/architecture.md).

Conteúdo do notebook:
- preparação dos dados e modelo original (baseline);
- algoritmo genético (representação dos indivíduos, fitness, seleção, cruzamento, mutação);
- 3 experimentos com configurações diferentes do GA;
- comparação entre modelo original e modelo otimizado;
- integração com a API da Anthropic (Claude) para interpretar os resultados,
  com prompt engineering e avaliação da qualidade das respostas (checagens
  heurísticas + LLM-as-judge).

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook tech_challenge_f02.ipynb
```

A integração com a LLM requer uma chave da Anthropic. Configure-a como variável de
ambiente antes de rodar o notebook — **nunca cole a chave diretamente no notebook**:

```bash
export ANTHROPIC_API_KEY="sua-chave-aqui"
```

Se estiver rodando no Google Colab, use o recurso de Secrets (ícone de chave na
barra lateral) com o nome `ANTHROPIC_API_KEY`.

**Erro `anthropic-workspace-id is required...`**: acontece quando a chave criada
no Claude Console é do tipo "identity-linked" com acesso a múltiplos workspaces.
Crie a chave em Settings → API keys escolhendo um workspace específico no
momento da criação — chaves de workspace único nunca exigem esse header.

## Testes

```bash
pytest tests/
```

Os testes cobrem as funções puras do algoritmo genético (`src/ga.py`), a
montagem do payload enviado à LLM (`src/llm.py`) e as checagens heurísticas de
qualidade (`src/quality.py`) — nenhum deles chama a API do Claude.

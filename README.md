# FIAP-TC-F02 — Tech Challenge Fase 2

Tech Challenge da Fase 2 da Pós Tech (IA para Devs) — **Projeto 1: Otimização de Modelos de Diagnóstico**.

O hospital universitário precisa melhorar a precisão e eficiência dos modelos de diagnóstico
desenvolvidos no Módulo 1. A solução usa um **algoritmo genético** para otimizar os
hiperparâmetros de um `RandomForestClassifier` e uma **LLM (Claude)** para gerar
interpretações em linguagem natural dos resultados para profissionais de saúde.

## Conteúdo

- [`tech_challenge_f02.ipynb`](tech_challenge_f02.ipynb) —
  notebook principal com:
  - preparação dos dados e modelo original (baseline);
  - algoritmo genético (representação dos indivíduos, fitness, seleção, cruzamento, mutação);
  - 3 experimentos com configurações diferentes do GA;
  - comparação entre modelo original e modelo otimizado;
  - integração com a API da Anthropic (Claude) para interpretar os resultados,
    com prompt engineering e avaliação da qualidade das respostas.

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

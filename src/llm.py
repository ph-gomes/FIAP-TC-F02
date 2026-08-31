"""Integração com a API da Anthropic (Claude) para interpretar as saídas do
Random Forest, gerar relatórios executivos e servir de juiz de qualidade.

O cliente `anthropic.Anthropic` é sempre recebido por parâmetro (`client`) em
vez de ser lido de uma variável global, para manter este módulo testável e
independente do notebook que o utiliza.
"""

import json
import time

import anthropic

from .logging_utils import logger, monitor_call

DEFAULT_MODEL = "claude-sonnet-5"

CLAUDE_SYSTEM_PROMPT = """Você é um assistente de apoio à decisão clínica que ajuda \
oncologistas a interpretar as saídas de um modelo de Machine Learning (Random Forest) \
treinado com dados tabulares (Wisconsin Breast Cancer Dataset) para classificar exames \
de câncer de mama como benignos ou malignos.

Regras obrigatórias:
- Baseie-se exclusivamente nos dados numéricos fornecidos no payload JSON. Não invente \
achados clínicos, características do exame ou informações do paciente que não estejam no payload.
- Nunca apresente a saída do modelo como um diagnóstico definitivo. Trate-a sempre como \
um indicador estatístico que deve ser validado por um profissional de saúde.
- Se o campo "clinical_notes" estiver vazio, não presuma conteúdo clínico adicional.
- Responda em português, em tom profissional e objetivo.
- Estruture SEMPRE a resposta nas seções: "Resumo", "Interpretação dos números", \
"Pontos de atenção", "Sugestão de próximos passos" e "Limitações"."""

JUDGE_SYSTEM_PROMPT = """Você é um avaliador técnico e imparcial de explicações \
médicas geradas por IA. Avalie a EXPLICAÇÃO fornecida com base APENAS nos DADOS \
originais fornecidos, atribuindo uma nota de 1 a 5 (5 = melhor) em cada critério:

- clareza: a explicação é fácil de entender para um profissional de saúde não-especialista em ML?
- fidelidade_aos_dados: a explicação usa somente informações presentes nos DADOS, sem inventar achados?
- utilidade_clinica: a explicação sugere próximos passos concretos e acionáveis?
- seguranca: a explicação evita apresentar o resultado como diagnóstico definitivo?

Responda ESTRITAMENTE em JSON, no formato:
{"clareza": <int>, "fidelidade_aos_dados": <int>, "utilidade_clinica": <int>, \
"seguranca": <int>, "justificativa": "<texto curto>"}"""


def build_case_payload(
    individuo,
    accuracy,
    recall,
    f1,
    prediction_probability,
    predicted_class,
    true_class=None,
    clinical_notes=None,
):
    """Monta o payload JSON enviado à LLM a partir das saídas do pipeline de Random Forest.

    `individuo` é a lista de hiperparâmetros escolhida pelo AG:
    [n_estimators, max_depth, min_samples_split, max_features].

    `clinical_notes` é um ponto de extensão opcional para anexar texto clínico
    real (laudo, anamnese) no futuro; por enquanto permanece opcional.
    """
    hiperparametros = {
        "n_estimators": individuo[0],
        "max_depth": individuo[1],
        "min_samples_split": individuo[2],
        "max_features": individuo[3],
    }

    return {
        "modelo": {
            "tipo": "Random Forest",
            "hiperparametros_escolhidos_pelo_ag": hiperparametros,
            "acuracia_conjunto_teste": round(float(accuracy), 4),
            "recall_maligno_conjunto_teste": round(float(recall), 4),
            "f1_score_conjunto_teste": round(float(f1), 4),
        },
        "caso": {
            "probabilidade_predita_maligno": round(float(prediction_probability), 4),
            "classe_predita": predicted_class,
            "classe_real_para_validacao_retrospectiva": true_class,
        },
        "clinical_notes": clinical_notes,
    }


@monitor_call("claude_explain_diagnosis")
def explain_diagnosis_with_claude(client, case_payload, model=DEFAULT_MODEL, max_tokens=700, max_retries=3):
    """Envia o payload de um caso ao Claude e retorna a explicação em linguagem natural.

    Implementa retentativa com backoff exponencial para lidar com limites de
    taxa (rate limiting) e indisponibilidades transitórias da API, cenário
    comum sob picos de demanda em produção.
    """
    user_message = (
        "Interprete o caso abaixo para apoiar a decisão clínica. "
        "Dados do caso (JSON):\n\n"
        f"{json.dumps(case_payload, ensure_ascii=False, indent=2)}"
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=CLAUDE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            explanation_parts = []
            for block in response.content:
                if isinstance(block, anthropic.types.TextBlock):
                    explanation_parts.append(block.text)

            if not explanation_parts:
                raise ValueError("Claude API response did not contain any TextBlock content.")

            return "".join(explanation_parts)
        except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
            last_error = exc
            wait_time = 2 ** attempt
            logger.info(
                f"Chamada à API do Claude falhou (tentativa {attempt}/{max_retries}), "
                f"aguardando {wait_time}s",
                extra={"extra_fields": {"error": str(exc), "attempt": attempt}},
            )
            time.sleep(wait_time)

    raise last_error


@monitor_call("claude_executive_summary")
def generate_executive_summary_with_claude(client, pipeline_results, model=DEFAULT_MODEL, max_tokens=10000, max_retries=3):
    """Gera um relatório executivo em Markdown a partir de um resumo (dict) do pipeline."""
    user_message = (
        "Escreva um relatório executivo, em português, para uma equipe clínica e técnica, "
        "resumindo os resultados do pipeline de otimização e classificação abaixo. "
        "Inclua os principais destaques, pontos de atenção e recomendações objetivas. "
        "Apresente o resumo como um texto contínuo, sem formatação específica de código ou JSON, adequado para um documento Markdown. "
        f"Dados (JSON):\n\n{json.dumps(pipeline_results, ensure_ascii=False, indent=2)}"
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=CLAUDE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            if response.stop_reason == "max_tokens":
                logger.error(
                    "Resposta truncada por max_tokens (thinking + texto combinados).",
                    extra={"extra_fields": {"stop_reason": response.stop_reason}},
                )
                raise ValueError("Response truncated at max_tokens before completing the report.")

            executive_summary_parts = []
            for block in response.content:
                if isinstance(block, anthropic.types.TextBlock):
                    executive_summary_parts.append(block.text)
                else:
                    logger.warning(
                        f"Bloco de conteúdo inesperado recebido do Claude para resumo: {type(block).__name__}",
                        extra={"extra_fields": {"unexpected_block_type": type(block).__name__, "block_content": str(block)}}
                    )

            if not executive_summary_parts:
                if not response.content:
                    logger.error(
                        "Claude API response.content estava vazio para o resumo executivo.",
                        extra={"extra_fields": {"raw_response_content": str(response.content)}}
                    )
                    raise ValueError("Claude API response.content estava vazio para o resumo executivo.")
                else:
                    logger.error(
                        "Claude API response.content continha blocos, mas nenhum era TextBlock para o resumo executivo.",
                        extra={"extra_fields": {"raw_response_content": str(response.content), "block_types": [type(b).__name__ for b in response.content]}}
                    )
                    raise ValueError("Claude API response did not contain any TextBlock content for the executive summary after inspecting blocks.")

            return "".join(executive_summary_parts)
        except (anthropic.RateLimitError, anthropic.APIStatusError, ValueError) as exc:
            last_error = exc
            wait_time = 2 ** attempt
            logger.info(
                f"Chamada à API do Claude para resumo falhou (tentativa {attempt}/{max_retries}), "
                f"aguardando {wait_time}s",
                extra={"extra_fields": {"error": str(exc), "attempt": attempt}},
            )
            time.sleep(wait_time)

    raise last_error


@monitor_call("claude_judge_explanation")
def judge_explanation_with_claude(client, case_payload, explanation_text, model=DEFAULT_MODEL, max_tokens=1024):
    user_message = (
        f"DADOS:\n{json.dumps(case_payload, ensure_ascii=False, indent=2)}\n\n"
        f"EXPLICAÇÃO A AVALIAR:\n{explanation_text}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text_parts = [
        block.text
        for block in response.content
        if isinstance(block, anthropic.types.TextBlock)
    ]

    if not text_parts:
        if response.stop_reason == "max_tokens":
            raise ValueError(
                "Claude atingiu max_tokens antes de gerar o texto do juiz "
                "(o thinking adaptativo pode ter consumido o orçamento todo); "
                "aumente max_tokens."
            )
        raise ValueError("Claude API response did not contain any TextBlock content for the judge.")

    raw_text = "".join(text_parts).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        logger.info(
            "Resposta do juiz não veio em JSON válido; devolvendo texto bruto.",
            extra={"extra_fields": {"raw_response": raw_text}},
        )
        return {"erro_parsing": True, "raw_response": raw_text}

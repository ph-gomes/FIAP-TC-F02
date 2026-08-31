"""Checagens heurísticas (determinísticas, sem LLM) sobre as explicações
geradas pelo Claude — primeira camada da avaliação de qualidade."""

FORBIDDEN_ABSOLUTE_CLAIMS = [
    "com certeza é",
    "definitivamente é câncer",
    "diagnóstico confirmado",
    "não há dúvida",
    "100% maligno",
    "100% benigno",
]

REQUIRED_SECTIONS = [
    "Resumo",
    "Interpretação dos números",
    "Pontos de atenção",
    "Sugestão de próximos passos",
    "Limitações",
]


def heuristic_quality_checks(explanation_text, case_payload):
    text_lower = explanation_text.lower()

    probability_pct = case_payload["caso"]["probabilidade_predita_maligno"] * 100
    cites_probability = (
        f"{case_payload['caso']['probabilidade_predita_maligno']:.2f}" in explanation_text
        or f"{probability_pct:.0f}" in explanation_text
        or f"{probability_pct:.1f}" in explanation_text
    )

    has_forbidden_claim = any(claim in text_lower for claim in FORBIDDEN_ABSOLUTE_CLAIMS)

    missing_sections = [s for s in REQUIRED_SECTIONS if s.lower() not in text_lower]

    reasonable_length = 200 <= len(explanation_text) <= 4000

    return {
        "cita_probabilidade_do_payload": cites_probability,
        "sem_afirmacao_absoluta_proibida": not has_forbidden_claim,
        "todas_secoes_presentes": len(missing_sections) == 0,
        "secoes_faltantes": missing_sections,
        "tamanho_dentro_do_esperado": reasonable_length,
    }

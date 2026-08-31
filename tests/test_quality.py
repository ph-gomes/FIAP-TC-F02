from src.quality import REQUIRED_SECTIONS, heuristic_quality_checks


def _payload(prob=0.87):
    return {"caso": {"probabilidade_predita_maligno": prob}}


def _explicacao_bem_formada():
    corpo = (
        "Resumo: caso com alta probabilidade de malignidade segundo o modelo.\n"
        "Interpretação dos números: a probabilidade estimada foi de 87%, "
        "com base nas métricas do Random Forest treinado.\n"
        "Pontos de atenção: recomenda-se investigação complementar.\n"
        "Sugestão de próximos passos: encaminhar para avaliação especializada.\n"
        "Limitações: trata-se de um indicador estatístico, não um diagnóstico definitivo.\n"
    )
    return corpo * 2  # garante tamanho dentro da faixa esperada (200-4000 caracteres)


def test_heuristic_checks_aprova_explicacao_bem_formada():
    checks = heuristic_quality_checks(_explicacao_bem_formada(), _payload())

    assert checks["cita_probabilidade_do_payload"] is True
    assert checks["sem_afirmacao_absoluta_proibida"] is True
    assert checks["todas_secoes_presentes"] is True
    assert checks["secoes_faltantes"] == []
    assert checks["tamanho_dentro_do_esperado"] is True


def test_heuristic_checks_detecta_afirmacao_absoluta_proibida():
    texto = ("Diagnóstico confirmado: 100% maligno, sem dúvidas. " * 6)

    checks = heuristic_quality_checks(texto, _payload())

    assert checks["sem_afirmacao_absoluta_proibida"] is False


def test_heuristic_checks_detecta_secoes_faltantes():
    texto = "Texto livre, sem nenhuma das seções exigidas pelo prompt. " * 5

    checks = heuristic_quality_checks(texto, _payload())

    assert checks["todas_secoes_presentes"] is False
    assert set(checks["secoes_faltantes"]) == set(REQUIRED_SECTIONS)


def test_heuristic_checks_detecta_texto_curto_demais():
    texto = "Resumo curto demais."

    checks = heuristic_quality_checks(texto, _payload())

    assert checks["tamanho_dentro_do_esperado"] is False

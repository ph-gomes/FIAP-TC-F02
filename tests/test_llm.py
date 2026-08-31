from src.llm import build_case_payload


def test_build_case_payload_monta_estrutura_esperada():
    individuo = [100, 8, 4, "sqrt"]

    payload = build_case_payload(
        individuo=individuo,
        accuracy=0.9561,
        recall=0.9302,
        f1=0.9411,
        prediction_probability=0.8734,
        predicted_class="Maligno",
        true_class="Maligno",
        clinical_notes=None,
    )

    assert payload["modelo"]["tipo"] == "Random Forest"
    assert payload["modelo"]["hiperparametros_escolhidos_pelo_ag"] == {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 4,
        "max_features": "sqrt",
    }
    assert payload["modelo"]["acuracia_conjunto_teste"] == 0.9561
    assert payload["caso"]["probabilidade_predita_maligno"] == 0.8734
    assert payload["caso"]["classe_predita"] == "Maligno"
    assert payload["caso"]["classe_real_para_validacao_retrospectiva"] == "Maligno"
    assert payload["clinical_notes"] is None


def test_build_case_payload_arredonda_metricas_numericas():
    payload = build_case_payload(
        individuo=[20, None, 2, "log2"],
        accuracy=0.123456,
        recall=0.654321,
        f1=0.999999,
        prediction_probability=0.111111,
        predicted_class="Benigno",
    )

    assert payload["modelo"]["acuracia_conjunto_teste"] == 0.1235
    assert payload["modelo"]["recall_maligno_conjunto_teste"] == 0.6543
    assert payload["modelo"]["f1_score_conjunto_teste"] == 1.0
    assert payload["caso"]["probabilidade_predita_maligno"] == 0.1111

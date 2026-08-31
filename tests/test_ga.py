import random

from sklearn.datasets import make_classification
import pandas as pd

from src.ga import (
    MAX_DEPTH_OPCOES,
    MAX_FEATURES_OPCOES,
    MIN_SAMPLES_SPLIT_OPCOES,
    N_ESTIMATORS_OPCOES,
    algoritmo_genetico,
    calcular_fitness,
    criar_individuo,
    cruzar,
    mutar,
    selecionar,
)


def test_criar_individuo_respeita_as_opcoes_definidas():
    individuo = criar_individuo()

    assert individuo[0] in N_ESTIMATORS_OPCOES
    assert individuo[1] in MAX_DEPTH_OPCOES
    assert individuo[2] in MIN_SAMPLES_SPLIT_OPCOES
    assert individuo[3] in MAX_FEATURES_OPCOES


def test_cruzar_preserva_genes_dos_dois_pais():
    random.seed(0)
    pai1 = [20, 2, 2, "sqrt"]
    pai2 = [150, None, 8, "log2"]

    filho = cruzar(pai1, pai2)

    assert len(filho) == 4
    for i, gene in enumerate(filho):
        assert gene == pai1[i] or gene == pai2[i]


def test_mutar_nao_altera_individuo_quando_taxa_e_zero():
    individuo = [20, 2, 2, "sqrt"]

    mutado = mutar(individuo, taxa_mutacao=0.0)

    assert mutado == individuo


def test_mutar_altera_no_maximo_um_gene_quando_taxa_e_um():
    random.seed(1)
    individuo = [20, 2, 2, "sqrt"]

    mutado = mutar(individuo, taxa_mutacao=1.0)

    diffs = [i for i in range(4) if mutado[i] != individuo[i]]
    assert len(diffs) <= 1


def test_mutar_nao_modifica_a_lista_original():
    individuo = [20, 2, 2, "sqrt"]

    mutar(individuo, taxa_mutacao=1.0)

    assert individuo == [20, 2, 2, "sqrt"]


def _dataset_sintetico():
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    metade = len(X) // 2
    return X.iloc[:metade], y.iloc[:metade], X.iloc[metade:], y.iloc[metade:]


def test_calcular_fitness_retorna_f1_valido():
    X_train, y_train, X_val, y_val = _dataset_sintetico()
    individuo = [20, 4, 2, "sqrt"]

    fitness = calcular_fitness(individuo, X_train, y_train, X_val, y_val)

    assert 0.0 <= fitness <= 1.0


def test_selecionar_retorna_metade_da_populacao():
    X_train, y_train, X_val, y_val = _dataset_sintetico()
    populacao = [criar_individuo() for _ in range(6)]

    selecionados = selecionar(populacao, X_train, y_train, X_val, y_val)

    assert len(selecionados) == 3
    assert all(individuo in populacao for individuo in selecionados)


def test_algoritmo_genetico_melhora_ou_mantem_o_fitness_entre_geracoes():
    X_train, y_train, X_val, y_val = _dataset_sintetico()

    _, _, historico_melhor, _ = algoritmo_genetico(
        tamanho_populacao=4,
        geracoes=3,
        taxa_mutacao=0.5,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=False,
    )

    assert len(historico_melhor) == 3
    # O melhor indivíduo é preservado entre gerações (elitismo), então o
    # melhor fitness nunca deve piorar de uma geração para a próxima.
    assert all(
        historico_melhor[i] <= historico_melhor[i + 1]
        for i in range(len(historico_melhor) - 1)
    )

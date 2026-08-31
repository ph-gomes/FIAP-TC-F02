"""Algoritmo genético usado para otimizar os hiperparâmetros do Random Forest.

Cada indivíduo é uma lista `[n_estimators, max_depth, min_samples_split, max_features]`.
"""

import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

N_ESTIMATORS_OPCOES = [20, 40, 60, 80, 100, 120, 150]
MAX_DEPTH_OPCOES = [2, 4, 6, 8, 10, 12, None]
MIN_SAMPLES_SPLIT_OPCOES = [2, 3, 4, 5, 6, 8]
MAX_FEATURES_OPCOES = ["sqrt", "log2", None]


def criar_individuo():
    return [
        random.choice(N_ESTIMATORS_OPCOES),
        random.choice(MAX_DEPTH_OPCOES),
        random.choice(MIN_SAMPLES_SPLIT_OPCOES),
        random.choice(MAX_FEATURES_OPCOES),
    ]


def calcular_fitness(individuo, X_train, y_train, X_val, y_val):
    modelo = RandomForestClassifier(
        n_estimators=individuo[0],
        max_depth=individuo[1],
        min_samples_split=individuo[2],
        max_features=individuo[3],
        random_state=42,
    )

    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_val)

    return f1_score(y_val, pred)


def selecionar(populacao, X_train, y_train, X_val, y_val):
    avaliados = []

    for individuo in populacao:
        fitness = calcular_fitness(individuo, X_train, y_train, X_val, y_val)
        avaliados.append([fitness, individuo])

    avaliados.sort(key=lambda x: x[0], reverse=True)

    metade = len(avaliados) // 2

    return [avaliados[i][1] for i in range(metade)]


def cruzar(pai1, pai2):
    ponto = random.randint(1, 3)

    return pai1[:ponto] + pai2[ponto:]


def mutar(individuo, taxa_mutacao):
    novo = individuo.copy()

    if random.random() < taxa_mutacao:
        gene = random.randint(0, 3)

        if gene == 0:
            novo[0] = random.choice(N_ESTIMATORS_OPCOES)
        elif gene == 1:
            novo[1] = random.choice(MAX_DEPTH_OPCOES)
        elif gene == 2:
            novo[2] = random.choice(MIN_SAMPLES_SPLIT_OPCOES)
        else:
            novo[3] = random.choice(MAX_FEATURES_OPCOES)

    return novo


def algoritmo_genetico(
    tamanho_populacao,
    geracoes,
    taxa_mutacao,
    X_train,
    y_train,
    X_val,
    y_val,
    verbose=True,
):
    populacao = [criar_individuo() for _ in range(tamanho_populacao)]

    historico_melhor = []
    historico_media = []

    melhor_individuo = None
    melhor_fitness = 0

    for geracao in range(geracoes):
        fitnesses = [
            calcular_fitness(individuo, X_train, y_train, X_val, y_val)
            for individuo in populacao
        ]

        melhor_da_geracao = max(fitnesses)
        media_da_geracao = sum(fitnesses) / len(fitnesses)

        indice_melhor = fitnesses.index(melhor_da_geracao)

        if melhor_da_geracao > melhor_fitness:
            melhor_fitness = melhor_da_geracao
            melhor_individuo = populacao[indice_melhor].copy()

        historico_melhor.append(melhor_da_geracao)
        historico_media.append(media_da_geracao)

        if verbose:
            print(
                "Geração", geracao + 1,
                "- Melhor:", round(melhor_da_geracao, 4),
                "- Média:", round(media_da_geracao, 4)
            )

        selecionados = selecionar(populacao, X_train, y_train, X_val, y_val)

        nova_populacao = [melhor_individuo.copy()]

        while len(nova_populacao) < tamanho_populacao:
            pai1 = random.choice(selecionados)
            pai2 = random.choice(selecionados)

            filho = cruzar(pai1, pai2)
            filho = mutar(filho, taxa_mutacao)

            nova_populacao.append(filho)

        populacao = nova_populacao

    return melhor_individuo, melhor_fitness, historico_melhor, historico_media

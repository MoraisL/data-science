"""Prepara o conjunto de dados para treinamento"""

import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """Prepara o conjunto de dados para treinamento na previsão"""

    # Calcula o número de exemplos.
    num_exemplos = data.shape[0]

    # Evita modificar os dados originais.
    data_processados = np.copy(data)

    # Normaliza o conjunto de dados.
    media_recursos = 0
    desvio_recursos = 0
    data_normalizados = data_processados
    if normalize_data:
        (
            data_normalizados,
            media_recursos,
            desvio_recursos
        ) = normalize(data_processados)

        # Substitui os dados processados pelos dados normalizados.
        # Precisamos dos dados normalizados abaixo ao adicionar polinômios e senoides.
        data_processados = data_normalizados

    # Adiciona características senoidais ao conjunto de dados.
    if sinusoid_degree > 0:
        senoides = generate_sinusoids(data_normalizados, sinusoid_degree)
        data_processados = np.concatenate((data_processados, senoides), axis=1)

    # Adiciona características polinomiais ao conjunto de dados.
    if polynomial_degree > 0:
        polinomios = generate_polynomials(data_normalizados, polynomial_degree, normalize_data)
        data_processados = np.concatenate((data_processados, polinomios), axis=1)

    # Adiciona uma coluna de uns a X.
    data_processados = np.hstack((np.ones((num_exemplos, 1)), data_processados))

    return data_processados, media_recursos, desvio_recursos

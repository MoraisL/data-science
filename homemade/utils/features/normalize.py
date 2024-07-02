"""Normalizar recursos"""

import numpy as np


def normalize(features):
    """Normaliza recursos.

    Normaliza os recursos de entrada X. Retorna uma versão normalizada de X onde o valor médio de
    cada recurso é 0 e a desvio é próximo de 1.

    :param features: conjunto de recursos.
    :return: conjunto de recursos normalizado.
    """

    # Copiar matriz original para evitar alterações.
    features_normalizados = np.copy(features).astype(float)

    # Obter valores médios para cada recurso (coluna) em X.
    features_media = np.mean(features, 0)

    # Calcular o desvio padrão para cada recurso.
    features_desvio = np.std(features, 0)

    # Subtrair os valores médios de cada recurso (coluna) de cada exemplo (linha)
    # para fazer com que todos os recursos sejam distribuídos ao redor de zero.
    if features.shape[0] > 1:
        features_normalizados -= features_media

    # Normalizar os valores de cada recurso para que todos os recursos fiquem próximos das fronteiras [-1:1].
    # Também evitar erro de divisão por zero.
    features_desvio[features_desvio == 0] = 1
    features_normalizados /= features_desvio

    return features_normalizados, features_media, features_desvio

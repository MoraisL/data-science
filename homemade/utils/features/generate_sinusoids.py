"""Adicionar recursos de senoide ao conjunto de recursos"""

import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """Estende o conjunto de dados com características de senoide.

    Retorna um novo array de características com mais recursos, compreendendo
    sin(x).

    :param dataset: conjunto de dados.
    :param sinusoid_degree: multiplicador para multiplicações de parâmetros senoidais
    """

    # Criar matriz de senoides.
    num_examples = dataset.shape[0]
    senoides = np.empty((num_examples, 0))

    # Gerar características senoidais do grau especificado.
    for grau in range(1, sinusoid_degree + 1):
        características_senoidais = np.sin(grau * dataset)
        senoides = np.concatenate((senoides, características_senoidais), axis=1)

    # Retornar características senoidais geradas.
    return senoides

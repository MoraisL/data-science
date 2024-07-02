"""Adicionar recursos polinomiais ao conjunto de recursos"""

import numpy as np
from .normalize import normalize


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """Estende o conjunto de dados com características polinomiais de um certo grau.

    Retorna um novo array de características com mais recursos, compreendendo
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.

    :param dataset: conjunto de dados para o qual queremos gerar polinômios.
    :param polynomial_degree: o grau máximo dos novos recursos.
    :param normalize_data: sinalizador que indica se os polinômios precisam ser normalizados ou não.
    """

    # Dividir características em dois conjuntos.
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    # Extrair parâmetros dos conjuntos.
    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    # Verificar se os dois conjuntos têm o mesmo número de linhas.
    if num_examples_1 != num_examples_2:
        raise ValueError('Não é possível gerar polinômios para dois conjuntos com número diferente de linhas')

    # Verificar se pelo menos um conjunto tem características.
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Não é possível gerar polinômios para dois conjuntos sem colunas')

    # Substituir conjunto vazio pelo não vazio.
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # Garantir que os conjuntos tenham o mesmo número de características para poder multiplicá-los.
    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    # Criar matriz de polinômios.
    polinomios = np.empty((num_examples_1, 0))

    # Gerar características polinomiais do grau especificado.
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            caracteristica_polinomial = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polinomios = np.concatenate((polinomios, caracteristica_polinomial), axis=1)

    # Normalizar polinômios se necessário.
    if normalize_data:
        polinomios = normalize(polinomios)[0]

    # Retornar características polinomiais geradas.
    return polinomios

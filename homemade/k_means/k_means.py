"""Módulo KMeans"""

import numpy as np

class KMeans:
    """Classe K-Means"""

    def __init__(self, data, num_clusters):
        """Construtor da classe K-Means.

        :param data: conjunto de dados de treinamento.
        :param num_clusters: número de clusters nos quais queremos dividir o conjunto de dados.
        """
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        """Função realiza o agrupamento de dados usando o algoritmo K-Means.

        :param max_iterations: número máximo de iterações de treinamento.
        """

        # Gera centróides aleatórios com base no conjunto de treinamento.
        centroids = KMeans.centroids_init(self.data, self.num_clusters)

        # Inicializa array padrão de IDs de centróides mais próximos.
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))

        # Executa o K-Means.
        for _ in range(max_iterations):
            # Encontra os centróides mais próximos para os exemplos de treinamento.
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)

            # Calcula as médias com base nos centróides mais próximos encontrados na parte anterior.
            centroids = KMeans.centroids_compute(
                self.data,
                closest_centroids_ids,
                self.num_clusters
            )

        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_init(data, num_clusters):
        """Inicializa num_clusters centróides que serão usados no K-Means no conjunto de dados X.

        :param data: conjunto de dados de treinamento.
        :param num_clusters: número de clusters nos quais queremos dividir o conjunto de dados.
        """

        # Obtém o número de exemplos de treinamento.
        num_examples = data.shape[0]

        # Reordena aleatoriamente os índices dos exemplos de treinamento.
        random_ids = np.random.permutation(num_examples)

        # Seleciona os primeiros K exemplos como centróides.
        centroids = data[random_ids[:num_clusters], :]

        # Retorna os centróides gerados.
        return centroids

    @staticmethod
    def centroids_find_closest(data, centroids):
        """Computa a pertinência do centróide para cada exemplo.

        Retorna os centróides mais próximos em closest_centroids_ids para um conjunto de dados X
        onde cada linha é um único exemplo. closest_centroids_ids é um vetor m x 1 de atribuições
        de centróides (ou seja, cada entrada está no intervalo [1..K]).

        :param data: conjunto de dados de treinamento.
        :param centroids: lista de pontos centróides.
        """

        # Obtém o número de exemplos de treinamento.
        num_examples = data.shape[0]

        # Obtém o número de centróides.
        num_centroids = centroids.shape[0]

        # Precisamos retornar as seguintes variáveis corretamente.
        closest_centroids_ids = np.zeros((num_examples, 1))

        # Itera sobre cada exemplo, encontra seu centróide mais próximo e armazena
        # o índice dentro de closest_centroids_ids na posição apropriada.
        # Concretamente, closest_centroids_ids(i) deve conter o índice do centróide
        # mais próximo ao exemplo i. Portanto, deve ser um valor no intervalo 1...num_centroids.
        for example_index in range(num_examples):
            distances = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):
                distance_difference = data[example_index, :] - centroids[centroid_index, :]
                distances[centroid_index] = np.sum(distance_difference ** 2)
            closest_centroids_ids[example_index] = np.argmin(distances)

        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clusters):
        """Calcula novos centróides.

        Retorna os novos centróides calculando as médias dos pontos de dados atribuídos a
        cada centróide.

        :param data: conjunto de dados de treinamento.
        :param closest_centroids_ids: lista de IDs de centróides mais próximos para cada exemplo de treinamento.
        :param num_clusters: número de clusters.
        """

        # Obtém o número de features.
        num_features = data.shape[1]

        # Precisamos retornar as seguintes variáveis corretamente.
        centroids = np.zeros((num_clusters, num_features))

        # Itera sobre cada centróide e calcula a média de todos os pontos que
        # pertencem a ele. Concretamente, o vetor de linha centroids(i, :)
        # deve conter a média dos pontos de dados atribuídos ao centróide i.
        for centroid_id in range(num_clusters):
            closest_ids = closest_centroids_ids == centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)

        return centroids

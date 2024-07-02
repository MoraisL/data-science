"""Módulo de Detecção de Anomalias"""

import math
import numpy as np

class GaussianAnomalyDetection:
    """Classe GaussianAnomalyDetection"""

    def __init__(self, data):
        """Construtor GaussianAnomalyDetection"""

        # Estima a distribuição Gaussiana.
        (self.mu_param, self.sigma_squared) = GaussianAnomalyDetection.estimate_gaussian(data)

        # Salva os dados de treinamento.
        self.data = data

    def multivariate_gaussian(self, data):
        """Calcula a função de densidade de probabilidade da distribuição gaussiana multivariada"""

        mu_param = self.mu_param
        sigma_squared = self.sigma_squared

        # Obtém o número de conjuntos de treinamento e características.
        (num_examples, num_features) = data.shape

        # Matriz de probabilidades.
        probabilities = np.ones((num_examples, 1))

        # Percorre todos os exemplos de treinamento e todas as características.
        for example_index in range(num_examples):
            for feature_index in range(num_features):
                # Calcula a potência de e.
                power_dividend = (data[example_index, feature_index] - mu_param[feature_index]) ** 2
                power_divider = 2 * sigma_squared[feature_index]
                e_power = -1 * power_dividend / power_divider

                # Calcula o multiplicador de prefixo.
                probability_prefix = 1 / math.sqrt(2 * math.pi * sigma_squared[feature_index])

                # Calcula a probabilidade para a característica atual do exemplo atual.
                probability = probability_prefix * (math.e ** e_power)
                probabilities[example_index] *= probability

        # Retorna as probabilidades para todos os exemplos de treinamento.
        return probabilities

    @staticmethod
    def estimate_gaussian(data):
        """Esta função estima os parâmetros de uma distribuição Gaussiana usando os dados em X."""

        # Obtém o número de características e exemplos.
        num_examples = data.shape[0]

        # Estima os parâmetros Gaussiano mu e sigma_squared para cada característica.
        mu_param = (1 / num_examples) * np.sum(data, axis=0)
        sigma_squared = (1 / num_examples) * np.sum((data - mu_param) ** 2, axis=0)

        # Retorna os parâmetros Gaussianos.
        return mu_param, sigma_squared

    @staticmethod
    def select_threshold(labels, probabilities):
        # pylint: disable=R0914
        """Encontra o melhor limiar (epsilon) para selecionar outliers"""

        best_epsilon = 0
        best_f1 = 0

        # Histórico de dados para construir os gráficos.
        precision_history = []
        recall_history = []
        f1_history = []

        # Calcula os passos do epsilon.
        min_probability = np.min(probabilities)
        max_probability = np.max(probabilities)
        step_size = (max_probability - min_probability) / 1000

        # Percorre todos os epsilons possíveis e escolhe aquele com a maior pontuação f1.
        for epsilon in np.arange(min_probability, max_probability, step_size):
            predictions = probabilities < epsilon

            # O número de falsos positivos: o rótulo verdadeiro diz que não é
            # uma anomalia, mas nosso algoritmo classificou incorretamente como uma anomalia.
            false_positives = np.sum((predictions == 1) & (labels == 0))

            # O número de falsos negativos: o rótulo verdadeiro diz que é uma anomalia,
            # mas nosso algoritmo classificou incorretamente como não sendo anômalo.
            false_negatives = np.sum((predictions == 0) & (labels == 1))

            # O número de verdadeiros positivos: o rótulo verdadeiro diz que é uma
            # anomalia e nosso algoritmo classificou corretamente como uma anomalia.
            true_positives = np.sum((predictions == 1) & (labels == 1))

            # Evita a divisão por zero.
            if (true_positives + false_positives) == 0 or (true_positives + false_negatives) == 0:
                continue

            # Precisão.
            precision = true_positives / (true_positives + false_positives)

            # Recall.
            recall = true_positives / (true_positives + false_negatives)

            # F1.
            f1_score = 2 * precision * recall / (precision + recall)

            # Salva dados históricos.
            precision_history.append(precision)
            recall_history.append(recall)
            f1_history.append(f1_score)

            if f1_score > best_f1:
                best_epsilon = epsilon
                best_f1 = f1_score

        return best_epsilon, best_f1, precision_history, recall_history, f1_history

"""Módulo de Regressão Linear"""

# Importar dependências.
import numpy as np
from ..utils.features import prepare_for_training

class LinearRegression:
    # pylint: disable=too-many-instance-attributes
    """Classe de Regressão Linear"""

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        # pylint: disable=too-many-arguments
        """Construtor de regressão linear.

        :param data: conjunto de treinamento.
        :param labels: saídas do conjunto de treinamento (valores corretos).
        :param polynomial_degree: grau de características polinomiais adicionais.
        :param sinusoid_degree: multiplicadores para características sinusoidais.
        :param normalize_data: sinalizador que indica se as características devem ser normalizadas.
        """

        # Normalizar características e adicionar coluna de uns.
        (
            data_processado,
            features_mean,
            features_deviation
        ) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processado
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # Inicializar parâmetros do modelo.
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, lambda_param=0, num_iterations=500):
        """Treina regressão linear.

        :param alpha: taxa de aprendizado (tamanho do passo para o gradiente descendente)
        :param lambda_param: parâmetro de regularização
        :param num_iterations: número de iterações de gradiente descendente.
        """

        # Executar gradiente descendente.
        cost_history = self.gradient_descent(alpha, lambda_param, num_iterations)

        return self.theta, cost_history

    def gradient_descent(self, alpha, lambda_param, num_iterations):
        """Gradiente descendente.

        Calcula os passos (deltas) que devem ser tomados para cada parâmetro theta a fim
        de minimizar a função de custo.

        :param alpha: taxa de aprendizado (tamanho do passo para o gradiente descendente)
        :param lambda_param: parâmetro de regularização
        :param num_iterations: número de iterações de gradiente descendente.
        """

        # Inicializar J_history com zeros.
        cost_history = []

        for _ in range(num_iterations):
            # Realizar um único passo de gradiente nos parâmetros theta.
            self.gradient_step(alpha, lambda_param)

            # Salvar o custo J em cada iteração.
            cost_history.append(self.cost_function(self.data, self.labels, lambda_param))

        return cost_history

    def gradient_step(self, alpha, lambda_param):
        """Passo de gradiente.

        Função realiza um passo de gradiente descendente para os parâmetros theta.

        :param alpha: taxa de aprendizado (tamanho do passo para o gradiente descendente)
        :param lambda_param: parâmetro de regularização
        """

        # Calcular o número de exemplos de treinamento.
        num_examples = self.data.shape[0]

        # Previsões da hipótese em todos os exemplos m.
        predictions = LinearRegression.hypothesis(self.data, self.theta)

        # A diferença entre previsões e valores reais para todos os exemplos m.
        delta = predictions - self.labels

        # Calcular parâmetro de regularização.
        reg_param = 1 - alpha * lambda_param / num_examples

        # Criar atalho para theta.
        theta = self.theta

        # Versão vetorial do gradiente descendente.
        theta = theta * reg_param - alpha * (1 / num_examples) * (delta.T @ self.data).T
        # Não devemos regularizar o parâmetro theta_zero.
        theta[0] = theta[0] - alpha * (1 / num_examples) * (self.data[:, 0].T @ delta).T

        self.theta = theta

    def get_cost(self, data, labels, lambda_param):
        """Obter o valor de custo para um conjunto de dados específico.

        :param data: conjunto de dados de treinamento ou teste.
        :param labels: saídas do conjunto de treinamento (valores corretos).
        :param lambda_param: parâmetro de regularização
        """

        data_processado = prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )[0]

        return self.cost_function(data_processado, labels, lambda_param)

    def cost_function(self, data, labels, lambda_param):
        """Função de custo.

        Mostra o quão preciso é nosso modelo com base nos parâmetros atuais do modelo.

        :param data: conjunto de dados de treinamento ou teste.
        :param labels: saídas do conjunto de treinamento (valores corretos).
        :param lambda_param: parâmetro de regularização
        """

        # Calcular o número de exemplos de treinamento e características.
        num_examples = data.shape[0]

        # Obter a diferença entre previsões e valores de saída corretos.
        delta = LinearRegression.hypothesis(data, self.theta) - labels

        # Calcular parâmetro de regularização.
        # Lembre-se de que não devemos regularizar o parâmetro theta_zero.
        theta_cut = self.theta[1:, 0]
        reg_param = lambda_param * (theta_cut.T @ theta_cut)

        # Calcular custo atual das previsões.
        custo = (1 / 2 * num_examples) * (delta.T @ delta + reg_param)

        # Vamos extrair o valor de custo da única célula da matriz de custo numpy.
        return custo[0][0]

    def predict(self, data):
        """Prever a saída para o conjunto de dados de entrada com base nos valores de theta treinados

        :param data: conjunto de características de treinamento.
        """

        # Normalizar características e adicionar coluna de uns.
        data_processado = prepare_for_training(
            data,
            self.polynomial_degree,
            self.sinusoid_degree,
            self.normalize_data,
        )[0]

        # Fazer previsões usando a hipótese do modelo.
        previsões = LinearRegression.hypothesis(data_processado, self.theta)

        return previsões

    @staticmethod
    def hypothesis(data, theta):
        """Função de hipótese.

        Prevê os valores de saída y com base nos valores de entrada X e nos parâmetros do modelo.

        :param data: conjunto de dados para o qual as previsões serão calculadas.
        :param theta: parâmetros do modelo.
        :return: previsões feitas pelo modelo com base no theta fornecido.
        """

        previsões = data @ theta

        return previsões

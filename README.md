# Ciência de dados
## Por: Vinícius de Morais Lino
# Guia de Estudo e Trilha de conhecimento em Ciência de Dados

Bem-vindo ao seu repositório de estudo sobre Ciência de Dados! Este guia fornecerá uma introdução rápida aos conceitos fundamentais de Análise de Dados e Machine Learning.

# Análise de Dados e Machine Learning

A análise de dados e o machine learning são áreas fundamentais da ciência de dados que permitem extrair insights valiosos e construir modelos preditivos a partir de conjuntos de dados.

## Análise de Dados

A análise de dados envolve o processo de inspecionar, limpar e modelar dados com o objetivo de descobrir informações úteis, sugerir conclusões e apoiar a tomada de decisões. Etapas comuns incluem:

- **Limpeza de Dados:** Identificação e correção de problemas nos dados, como valores ausentes ou inconsistentes.
  
- **Exploração de Dados:** Compreensão das características dos dados através de estatísticas descritivas e visualizações.

- **Transformação de Dados:** Preparação dos dados para análise, incluindo normalização, codificação de variáveis categóricas, entre outros.

## Machine Learning

Machine learning é uma disciplina que utiliza algoritmos e modelos estatísticos para permitir que sistemas computacionais aprendam com dados e façam previsões ou decisões sem serem explicitamente programados. Principais conceitos incluem:

- **Tipos de Aprendizado:** Supervisionado, não supervisionado e por reforço.
  
- **Modelagem de Dados:** Desenvolvimento e avaliação de modelos preditivos, como regressão, classificação e agrupamento.

- **Avaliação de Modelos:** Métricas como precisão, recall, e curvas ROC para avaliar a performance dos modelos.

Ambas as áreas são essenciais para transformar dados em insights acionáveis e automatizar processos de tomada de decisão através de técnicas avançadas de análise e aprendizado de máquina.

## Sumário

1. [Instalação do Python](#instalação-do-python)
2. [Bibliotecas Essenciais](#bibliotecas-essenciais)
3. [Pandas](#pandas)
4. [NumPy](#numpy)
5. [Matplotlib](#matplotlib)
6. [K-Means](#k-means)
7. [Detecção de Anomalias](#detecção-de-anomalias)
8. [Regressão Linear](#regressão-linear)
9. [Redes Neurais](#redes-neurais)

## Instalação do Python

1. Acesse o [site oficial do Python](https://www.python.org/).
2. Baixe e instale o Python para o seu sistema operacional.
3. Verifique a instalação abrindo um terminal e digitando `python --version`.

## Bibliotecas Essenciais

Instale as bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib scikit-learn
```


### Pandas
Pandas é uma biblioteca essencial para manipulação e análise de dados. Ela fornece estruturas de dados flexíveis e intuitivas, como DataFrames.
```
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())
print(df.describe())
```

### NumPy
NumPy é uma biblioteca para operações matemáticas em arrays. Ela é a base para muitas outras bibliotecas em Python.
```
import numpy as np

arr = np.array([1, 2, 3, 4])
print(np.mean(arr))
print(np.std(arr))

```

### Matplotlib
Matplotlib é uma biblioteca para criação de visualizações em gráficos.
```
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Exemplo de Gráfico')
plt.show()

```
## Cálculos Importantes em Análise de Dados
Alguns cálculos são fundamentais em Análise de Dados, incluindo:

Média: np.mean(data)
Mediana: np.median(data)
Desvio Padrão: np.std(data)
Correlação: np.corrcoef(data)


### K-Means
K-Means é um algoritmo de clustering usado para agrupar pontos de dados em clusters. Ele funciona atribuindo pontos de dados a um cluster com base na proximidade média dos pontos de dados dentro do cluster. Exemplo:
```
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

```

### Regressão Linear
A regressão linear é uma técnica usada para prever o valor de uma variável com base em outra variável. Exemplo:

```
from sklearn.linear_model import LinearRegression

# Dados de exemplo
X = np.array([[1], [2], [3], [4]])
y = np.array([10, 20, 25, 30])

# Aplicando regressão linear
model = LinearRegression().fit(X, y)

# Fazendo previsões
predictions = model.predict(X)
print(predictions)

```


### Redes Neurais
Redes Neurais são modelos computacionais inspirados no cérebro humano, usados para reconhecer padrões complexos. Exemplo:

```
from sklearn.neural_network import MLPClassifier

# Dados de exemplo
X = [[0., 0.], [1., 1.]]
y = [0, 1]

# Treinando uma rede neural
clf = MLPClassifier(random_state=1, max_iter=300).fit(X, y)

# Fazendo previsões
predictions = clf.predict([[2., 2.], [-1., -2.]])
print(predictions)

```

## Machine Learning Map

![Machine Learning Map](images/machine-learning-map.png)

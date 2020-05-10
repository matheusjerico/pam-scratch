# Tarefa 2: Aprendizado Não-Supervisionado

**Autor**: Matheus Jericó Palhares <br>
**LinkedIn**: https://linkedin.com/in/matheusjerico <br>
**Github**: https://github.com/matheusjerico

### 1) Implementar a função “fit_pam(pontos, alpha)”, com K fixo e igual a 3. Retorne os centroids finais.

**Variáveis**:
- pontos: conjunto de pontos 2D (casos x mortes) que serão clusterizados
- alpha: valor de α que indica a quantidade de tentativas sem melhoria de custo total médio que devem ser realizadas antes da interrupção do algoritmo



O relatório será o notebook exportado para um arquivo HTML e deve conter:
- Um scatter plot mostrando os medoids (com marcador x) e seus respectivos pontos (cada cluster deve estar em uma cor distinta)
- Para cada cluster, também devem ser exibidos seus custos totais, bem como o custo total médio
- Discorra sobre cada cluster: o que eles indicam?



#### 1. Bibliotecas


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from copy import deepcopy
```

#### 2. Carregando dados


```python
dataset = pd.read_csv("./Dataset/base-covid-19-us.csv")
dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county</th>
      <th>cases</th>
      <th>deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbeville</td>
      <td>84</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Acadia</td>
      <td>741</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Accomack</td>
      <td>116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ada</td>
      <td>4264</td>
      <td>41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adair</td>
      <td>325</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.shape
```




    (1570, 3)




```python
dataset.sort_values(by=['deaths'], inplace=True)
```


```python
cases = dataset['cases'].values
deaths = dataset['deaths'].values
X = np.array(list(zip(cases, deaths)))
plt.scatter(cases, deaths, c='black', s=10)
```




    <matplotlib.collections.PathCollection at 0x7f6f1c2e81d0>




![png](output_10_1.png)


#### 3. Criando classe do Pam


```python
class Pam:
    def __init__(self, k, pontos, alpha = 100):
        self.k = k
        self.alpha = alpha
        self.data = pontos
        
    # Calculando custo
    def custo(self, a, b):
      retorno = (a[0] - b[0])**2 + (a[1] - b[1])**2
      return retorno
    
    def fit(self):
        self.medoids = []
        self.mse = -1
        self.custos_totais = []

        medoids_tmp = []
        indices_para_escolher = list(range(len(self.data)))
        indices = np.random.randint(0, len(self.data), size=self.k)
        tentativas = 0

        for indice in indices:
            indices_para_escolher.remove(indice)
            self.medoids.append(self.data[indice])
        
        # armazenando o valor dos medoids quando ele for atualizado
        indices_tmp = []
        
        # label dos clusters 
        self.clusters = np.zeros(len(self.data))

        custos_totais_tmp = [0] * self.k
        
        # Loop de treinamento
        for i in range(len(self.data) - self.k):
            # atribuindo cada valor ao cluster mais próximo
            for j in range(len(self.data)):
                if j in indices:
                    continue

                custos = []
                for medoid in self.medoids:
                    custos.append(self.custo(self.data[j], medoid))
                cluster = np.argmin(custos)
                self.clusters[j] = cluster
                custos_totais_tmp[cluster] += min(custos)

            # calculando o custo total medio
            mse_tmp = 0
            unique, counts = np.unique(self.clusters, return_counts=True)
            for j in range(self.k):
                if counts[j] > 0:
                    custos_totais_tmp[j] += custos_totais_tmp[j]/counts[j]

            mse_tmp = np.mean(custos_totais_tmp)

            if (self.mse == -1) | (self.mse > mse_tmp):
                self.mse = mse_tmp
                self.custos_totais = deepcopy(custos_totais_tmp)
                # guardando valores dos indices dos medoids antigos
                indices_tmp = deepcopy(indices)
                tentativas = 0
            else:
                tentativas += 1

            if tentativas >= self.alpha:
                break

            # encontrando novo medoid
            posicao = np.random.randint(0, len(indices_para_escolher)) 
            indice = indices_para_escolher[posicao]
            indices_para_escolher.remove(indice)
            custos = []
            for medoid in self.medoids:
                custos.append(self.custo(self.data[indice], medoid))
            new_medoid_index = np.argmin(custos)
            indices[new_medoid_index] = indice
            self.medoids[new_medoid_index] = self.data[indice]

    def plot(self):
        colors = int(self.k/6 + 1)*['r', 'g', 'b', 'y', 'c', 'm']
        fig, ax = plt.subplots()

        for i in range(self.k):
            points = np.array([self.data[j] for j in range(len(self.data)) if self.clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
            for medoid in self.medoids:
                ax.scatter(medoid[0], medoid[1], marker='X', s=100, c='black')
```

#### 4. Inicializando a classe com os parametros desejados do exercício e realizando o treinamento


```python
pam = Pam(k = 3, pontos = X, alpha = 100)
pam.fit()
pam.plot()
```


![png](output_14_0.png)



```python
pam.medoids
```




    [array([39302,  1106]), array([87378,  3236]), array([70,  0])]




```python
dataset['cluster'] = pam.clusters
```


```python
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1570 entries, 0 to 995
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   county   1570 non-null   object 
     1   cases    1570 non-null   int64  
     2   deaths   1570 non-null   int64  
     3   cluster  1570 non-null   float64
    dtypes: float64(1), int64(2), object(1)
    memory usage: 61.3+ KB



```python
dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county</th>
      <th>cases</th>
      <th>deaths</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abbeville</td>
      <td>84</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>865</th>
      <td>Manitowoc</td>
      <td>17</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>866</th>
      <td>Marathon</td>
      <td>108</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>869</th>
      <td>Maries</td>
      <td>20</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>871</th>
      <td>Marinette</td>
      <td>27</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 5. Visualizando graficamente


```python
pam.plot()
```


![png](output_20_0.png)



```python
pam.medoids
```




    [array([39302,  1106]), array([87378,  3236]), array([70,  0])]



**Resposta**: <br>
Os clusters representam os estados com base na quantidade de casos e mortes do COVID-19.
- O primeiro cluster (verde), representa os estados que tiveram baixa quantidade de casos e mortes devido ao COVID-19
- O segundo cluster(azul), representa os estados que tiveram quantidade moderada/alta de casos e mortes devido ao COVID-19.
- O terceiro cluster (vermelho), representa o estado de New York (possuem uma quantidade muito elevada de casos de COVID-19


#### 6. Para cada cluster, também devem ser exibidas as distâncias médias entre os pontos e seu respectivo centróide final


```python
def print_custos(pam):
    for i in range(pam.k):  
        print("Custo Total entre os pontos do medoid {}: {}".format(i, pam.custos_totais[i]))

    print("Custo Total Médio: {}".format(pam.mse))
```


```python
print_custos(pam)
```

    Custo Total entre os pontos do medoid 0: 3240794.6790697672
    Custo Total entre os pontos do medoid 1: 938980649735.3953
    Custo Total entre os pontos do medoid 2: 15978515.073991032
    Custo Total Médio: 312999956348.38275


### **Obs**: Retirando o estado de New York, a visualização do cluster fica muito melhor.

#### 7. Retirando o estado de New York
- Retirar o estado de New York, utilizar o algoritmo Kmeans e visualizar os cluster.


```python
dataset=dataset.iloc[:-1,:]
```


```python
cases = dataset['cases'].values
deaths = dataset['deaths'].values
X = np.array(list(zip(cases, deaths)))
plt.scatter(cases, deaths, c='black', s=10)
```




    <matplotlib.collections.PathCollection at 0x7f6f1c127518>




![png](output_29_1.png)



```python
pam = Pam(k = 3, pontos = X, alpha = 100)
pam.fit()
```


```python
pam.plot()
```


![png](output_31_0.png)



```python
pam.medoids
```




    [array([352,   9]), array([27995,   479]), array([8600,  371])]




```python
dataset['cluster'] = pam.clusters
```


```python
print_custos(pam)
```

    Custo Total entre os pontos do medoid 0: 811789.5645161291
    Custo Total entre os pontos do medoid 1: 184454710791.77243
    Custo Total entre os pontos do medoid 2: 1089847.656140351
    Custo Total Médio: 61485537476.331024


# Desafios

#### Plotar o gráfico que permite visualizar o elbow point, variando o valor de K e indicar qual o melhor valor


```python
def elbow_plot(data, k_max):
    dataset = data.copy()
    custos = []

    cases = dataset['cases'].values
    deaths = dataset['deaths'].values
    X = np.array(list(zip(cases, deaths)))

    for i in range(k_max - 2):
        pam = Pam(k = i+2, pontos = X)
        pam.fit()

        dataset['cluster'] = pam.clusters
        
        custos.append([i+2, pam.mse])

    x = []
    y = []

    fig, ax = plt.subplots()
    for i in range(len(custos)):
        x.append(custos[i][0])
        y.append(custos[i][1])

    ax.set_xlabel('Quantidade de Clusters')
    ax.set_ylabel('Distância Média')

    ax.plot(x, y, c='b', marker='x')
```


```python
elbow_plot(dataset, 8)
```


![png](output_38_0.png)


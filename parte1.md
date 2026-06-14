# AV3 - Parte 1 - Otimização Contínua com Meta-Heurísticas

## Objetivo

Implementar três algoritmos de busca/otimização meta-heurística:

* Hill Climbing (HC)
* Local Random Search (LRS)
* Global Random Search (GRS)

para resolver os 6 problemas de otimização contínua definidos no enunciado.

---

# Arquivos de referência

Os seguintes arquivos disponibilizados pelo professor devem ser utilizados como base conceitual e estrutural:

* `busca.py`
* `busca_discreta.py`
* `algoritmos_geneticos.py`
* `exemplo_ga.py`

---

# Estrutura sugerida

```text
src/
│
├── functions.py
├── algorithms/
│   ├── hill_climbing.py
│   ├── local_random_search.py
│   └── global_random_search.py
│
├── core/
│   ├── problem.py
│   ├── result.py
│   └── experiment.py
│
├── experiments/
│   ├── experiment_f1.py
│   ├── experiment_f2.py
│   ├── experiment_f3.py
│   ├── experiment_f4.py
│   ├── experiment_f5.py
│   └── experiment_f6.py
│
└── main.py
```

---

# Classe Problem

Criar uma abstração para representar um problema de otimização.

Exemplo:

```python
class Problem:
    def __init__(
        self,
        name,
        objective_function,
        bounds,
        maximize
    ):
        ...
```

Campos:

* name
* objective_function
* bounds
* maximize

Exemplo:

```python
Problem(
    name="Sphere",
    objective_function=f1,
    bounds=[(-100,100), (-100,100)],
    maximize=False
)
```

---

# Critério de Caixa (Bounds)

Todo candidato gerado deve ser validado.

Antes da avaliação:

```python
lower <= x <= upper
```

Caso ultrapasse o limite:

* descartar candidato
  ou
* fazer clipping

```python
x = max(lower, min(x, upper))
```

Preferir clipping.

---

# Critério de Parada

Todos os algoritmos devem possuir:

## Parada principal

```python
MAX_ITERATIONS = 1000
```

## Parada antecipada

Encerrar quando não houver melhoria por:

```python
STALL_LIMIT = 100
```

iterações consecutivas.

---

# Execuções Independentes

Cada algoritmo deve ser executado:

```python
R = 100
```

vezes para cada função.

Cada execução gera:

```python
x_best
f(x_best)
```

Armazenar todos os resultados.

---

# Estatísticas Necessárias

Para cada combinação:

```text
Algoritmo × Função
```

calcular:

* melhor solução
* pior solução
* média
* desvio padrão
* moda

A moda será utilizada como resultado final solicitado pelo professor.

---

# Hill Climbing

## Requisitos

Ponto inicial:

```python
x = limite_inferior
```

para todas as dimensões.

Exemplo:

```python
[-100, -100]
```

para a função 1.

---

## Vizinhança

Utilizar:

```python
|x - y| <= epsilon
```

onde:

```python
epsilon = 0.1
```

Inicialmente.

Gerar perturbação uniforme:

```python
y = x + U(-epsilon, epsilon)
```

---

## Estudo do epsilon

Realizar busca experimental para identificar o menor valor de epsilon capaz de encontrar o ótimo.

Exemplo:

```python
0.5
0.2
0.1
0.05
0.01
```

---

# Local Random Search (LRS)

## Inicialização

Gerar candidato inicial:

```python
x ~ Uniform(bounds)
```

---

## Geração de vizinhos

Utilizar distribuição normal:

```python
y = x + N(0, sigma)
```

---

## Hiperparâmetro

Testar valores:

```python
sigma = [
    1.0,
    0.5,
    0.25,
    0.1,
    0.05,
    0.01
]
```

Identificar o menor sigma que encontra o ótimo.

---

# Global Random Search (GRS)

## Estratégia

A cada iteração gerar:

```python
y ~ Uniform(bounds)
```

independentemente da solução atual.

---

## Atualização

Se y for melhor que x_best:

```python
x_best = y
```

---

# Funções do Trabalho

## F1 - Sphere

Minimização

```python
f(x1,x2) = x1² + x2²
```

Domínio:

```python
[-100,100]
[-100,100]
```

Ótimo esperado:

```python
(0,0)
```

---

## F2

Maximização

```python
f(x1,x2)=
exp(-(x1²+x2²))
+
2*exp(-((x1-1.7)**2+(x2-1.7)**2))
```

Domínio:

```python
x1 ∈ [-2,4]
x2 ∈ [-2,5]
```

---

## F3 - Ackley

Minimização

Domínio:

```python
[-8,8]
[-8,8]
```

Ótimo esperado:

```python
(0,0)
```

---

## F4 - Rastrigin

Minimização

Domínio:

```python
[-5.12,5.12]
```

Ótimo esperado:

```python
(0,0)
```

---

## F5

Maximização

Domínio:

```python
[-10,10]
```

---

## F6

Maximização

Domínio:

```python
[-1,3]
```

---

# Resultado Esperado

Gerar uma tabela para cada função.

Exemplo:

| Algoritmo | Moda | Média | Melhor |
| --------- | ---- | ----- | ------ |
| HC        | ...  | ...   | ...    |
| LRS       | ...  | ...   | ...    |
| GRS       | ...  | ...   | ...    |

---

# Visualizações

Gerar:

* gráfico de convergência;
* boxplot das 100 execuções;
* histograma dos resultados.

---

# Observação Importante

Não utilizar bibliotecas que implementem diretamente:

* Hill Climbing
* Random Search
* Meta-Heurísticas prontas

É permitido utilizar:

* numpy
* matplotlib
* pandas
* scipy (estatísticas)

desde que os algoritmos sejam implementados manualmente.

# AV3 - Parte 2.2 - Algoritmo Genético para o Problema do Caixeiro Viajante 3D

## Objetivo

Implementar um Algoritmo Genético (Genetic Algorithm - GA) para encontrar uma rota subótima para um drone que deve:

1. Partir da origem.
2. Visitar todos os pontos fornecidos.
3. Retornar à origem.
4. Minimizar a distância total percorrida.

---

# Contexto do Problema

O professor disponibilizou um arquivo:

```text
CaixeiroGrupos.csv
```

contendo pontos tridimensionais.

Cada ponto representa uma posição que deve ser visitada pelo drone.

---

# Tipo de Problema

Problema clássico de otimização combinatória.

Objetivo:

```python
minimizar(distancia_total)
```

---

# Representação do Cromossomo

Cada indivíduo representa uma rota.

Exemplo:

```python
[4, 12, 7, 2, 18, 1, 10]
```

Significa:

```text
Origem
→ P4
→ P12
→ P7
→ P2
→ P18
→ P1
→ P10
→ Origem
```

---

# Restrição Fundamental

Não pode existir repetição de pontos.

Exemplo inválido:

```python
[4, 7, 2, 7, 10]
```

Ponto 7 repetido.

---

# Estrutura Recomendada

```text
src3/
│
├── tsp/
│   ├── point.py
│   ├── route.py
│   ├── fitness.py
│   ├── crossover.py
│   ├── mutation.py
│   ├── selection.py
│   └── population.py
│
├── ga/
│   └── genetic_algorithm.py
│
├── data/
│   └── CaixeiroGrupos.csv
│
└── main.py
```

---

# Carregamento dos Dados

Criar um módulo responsável por:

```python
load_points(csv_path)
```

Retorno:

```python
[
    (x,y,z),
    (x,y,z),
    ...
]
```

---

# Quantidade de Pontos

O professor solicita:

```python
30 < Npontos < 60
```

Escolha recomendada:

```python
Npontos = 50
```

Justificativa:

* suficientemente complexo;
* custo computacional viável.

---

# Estrutura do Indivíduo

Exemplo:

```python
class Individual:
    chromosome
    fitness
```

---

# Função de Distância

Utilizar distância euclidiana 3D.

```python
d =
sqrt(
    (x2-x1)^2 +
    (y2-y1)^2 +
    (z2-z1)^2
)
```

---

# Função Custo

Somar:

1. Origem → primeiro ponto
2. Todos os pares consecutivos
3. Último ponto → origem

---

# Exemplo

```python
Ori
→ P4
→ P1
→ P5
→ P3
→ P2
→ Ori
```

Fitness:

```python
d(Ori,P4)
+
d(P4,P1)
+
d(P1,P5)
+
d(P5,P3)
+
d(P3,P2)
+
d(P2,Ori)
```

---

# Fitness

Como queremos minimizar:

```python
custo = distancia_total
```

Pode-se utilizar:

```python
fitness = 1/(1+custo)
```

para converter em maximização.

---

# População Inicial

Gerar rotas aleatórias.

Exemplo:

```python
random.shuffle(lista_pontos)
```

---

# Hiperparâmetros

Valores iniciais sugeridos:

```python
POPULATION_SIZE = 100
```

```python
MAX_GENERATIONS = 500
```

```python
CROSSOVER_RATE = 0.9
```

```python
MUTATION_RATE = 0.01
```

---

# Seleção

## Método obrigatório

Torneio.

---

# Torneio

Selecionar:

```python
k = 3
```

indivíduos aleatórios.

Escolher:

```python
melhor fitness
```

---

# Recombinação

O professor exige uma adaptação do crossover de dois pontos para permutações.

Não pode gerar cromossomos inválidos.

---

# Operador Recomendado

Order Crossover (OX)

Implementação compatível com o enunciado.

---

# Exemplo

Pai A

```python
[1 2 3 4 5 6 7 8]
```

Pai B

```python
[5 7 1 8 3 6 2 4]
```

Pontos:

```python
c1=3
c2=5
```

Copiar trecho:

```python
[4 5 6]
```

Completar com genes restantes do segundo pai.

Resultado:

```python
[7 1 4 5 6 2 8 3]
```

---

# Validação

Após crossover:

```python
len(set(chromosome))
==
len(chromosome)
```

deve ser verdadeiro.

---

# Mutação

Obrigatória:

```python
1%
```

---

# Operador de Mutação

Swap Mutation.

---

# Exemplo

Antes:

```python
[1,2,3,4,5]
```

Seleciona:

```python
i=1
j=4
```

Depois:

```python
[1,5,3,4,2]
```

---

# Elitismo

O professor pede análise da necessidade.

Implementar elitismo opcional.

---

# Valor Recomendado

```python
ELITE_SIZE = 2
```

ou

```python
ELITE_SIZE = 5
```

---

# Critério de Parada

Parar quando:

```python
generation >= MAX_GENERATIONS
```

ou

```python
não houver melhoria
```

por:

```python
50 gerações
```

---

# Estatísticas

Registrar por geração:

```python
best_fitness
```

```python
average_fitness
```

```python
worst_fitness
```

---

# Resultados Solicitados

Determinar:

## Moda das gerações

Número de gerações necessário para atingir solução aceitável.

---

# Experimento

Executar:

```python
100 execuções
```

independentes.

Registrar:

```python
gerações até convergência
```

---

# Tabela Esperada

| Execução | Gerações | Distância Final |
| -------- | -------- | --------------- |
| 1        | ...      | ...             |
| 2        | ...      | ...             |
| ...      | ...      | ...             |

---

# Tabela Resumo

| Métrica          | Valor |
| ---------------- | ----- |
| Melhor distância | ...   |
| Pior distância   | ...   |
| Média            | ...   |
| Moda             | ...   |
| Desvio padrão    | ...   |

---

# Gráficos

Gerar:

## Convergência

```text
geração × melhor fitness
```

---

## Fitness médio

```text
geração × fitness médio
```

---

## Histograma

```text
distribuição das gerações
```

---

# Comparação com e sem Elitismo

Executar:

```python
ELITE_SIZE = 0
```

e

```python
ELITE_SIZE = 5
```

Comparar:

* convergência;
* qualidade final;
* tempo de execução.

---

# Critério de Sucesso

O algoritmo deve produzir:

* rota válida;
* sem repetição de pontos;
* retorno à origem;
* distância significativamente menor que uma rota aleatória.

---

# Bibliotecas Permitidas

* numpy
* pandas
* matplotlib
* random
* math
* time

---

# Bibliotecas Não Permitidas

Não utilizar implementações prontas de:

* Genetic Algorithm
* TSP Solver
* Evolutionary Algorithms

Toda a lógica genética deve ser implementada manualmente.

# AV3 - Parte 2 - Simulated Annealing para o Problema das 8 Rainhas

## Objetivo

Implementar o algoritmo Simulated Annealing (Têmpera Simulada) para resolver o problema das 8 Rainhas.

O algoritmo deve:

1. Encontrar uma solução válida.
2. Continuar executando até encontrar as 92 soluções distintas do problema.
3. Avaliar o custo computacional necessário para encontrar todas as soluções.

---

# Problema

Posicionar 8 rainhas em um tabuleiro 8x8 de forma que nenhuma rainha ataque outra.

Cada rainha ocupa uma coluna fixa.

Assim, uma solução candidata possui:

```python
x = [r1, r2, r3, r4, r5, r6, r7, r8]
```

onde:

```python
ri = linha ocupada pela rainha da coluna i
```

Exemplo:

```python
[5,1,4,2,6,1,4,7]
```

significa:

* coluna 1 -> linha 5
* coluna 2 -> linha 1
* ...
* coluna 8 -> linha 7

---

# Representação do Estado

Cada posição do vetor representa:

```python
indice = coluna
valor = linha
```

Domínio:

```python
linha ∈ [1,8]
```

Exemplo válido:

```python
[4,2,7,3,6,8,5,1]
```

Não é necessário impedir linhas repetidas.

O algoritmo deve aprender a evitá-las através da função objetivo.

---

# Função Objetivo

Conforme sugerido pelo professor:

```python
f(x) = 28 - h(x)
```

onde:

```python
h(x)
```

é o número de pares de rainhas que se atacam.

---

# Quantidade Máxima de Pares

Para 8 rainhas:

```python
C(8,2) = 28
```

Logo:

```python
0 <= h(x) <= 28
```

---

# Melhor Solução Possível

Quando:

```python
h(x) = 0
```

temos:

```python
f(x) = 28
```

Portanto:

## Problema de Maximização

```python
maximizar f(x)
```

Objetivo:

```python
f(x) = 28
```

---

# Cálculo dos Ataques

Duas rainhas se atacam quando:

## Mesma linha

```python
ri == rj
```

## Mesma diagonal

```python
abs(ri-rj) == abs(i-j)
```

---

# Exemplo

```python
for i in range(8):
    for j in range(i+1,8):
        ...
```

Cada conflito incrementa:

```python
h += 1
```

---

# Estrutura do Projeto

```text
src2/
│
├── queens/
│   ├── board.py
│   ├── objective.py
│   ├── neighborhood.py
│   └── solution_repository.py
│
├── annealing/
│   ├── simulated_annealing.py
│   └── cooling.py
│
├── experiments/
│   └── run_queens.py
│
└── main.py
```

---

# Solução Inicial

Gerar aleatoriamente:

```python
x = [
    randint(1,8)
    for _ in range(8)
]
```

---

# Temperatura Inicial

Valor sugerido:

```python
T0 = 100
```

Também testar:

```python
10
50
100
500
```

e registrar resultados.

---

# Função de Perturbação

IMPORTANTE:

O professor pede uma perturbação local, sem acesso ao espaço inteiro.

Não gerar um estado completamente novo.

---

# Operador de Vizinhança

Escolher aleatoriamente:

```python
coluna
```

e alterar apenas uma rainha.

Exemplo:

Estado atual:

```python
[4,2,7,3,6,8,5,1]
```

Seleciona coluna 3.

Novo estado:

```python
[4,2,5,3,6,8,5,1]
```

Apenas uma posição muda.

---

# Aceitação

Se melhorar:

```python
Δ > 0
```

aceitar.

---

# Aceitação Probabilística

Se piorar:

```python
P = exp(Δ/T)
```

onde:

```python
Δ = f(novo) - f(atual)
```

---

# Critério

```python
if random() < P:
    aceita
```

---

# Resfriamento

Utilizar resfriamento geométrico.

```python
T = alpha * T
```

---

# Valor Inicial

```python
alpha = 0.99
```

Testar:

```python
0.95
0.97
0.99
0.995
```

---

# Critérios de Parada

Parar quando:

## Critério 1

```python
f(x) == 28
```

Encontrou solução válida.

---

## Critério 2

```python
iteracoes >= MAX_ITER
```

---

# Armazenamento de Soluções

Criar estrutura:

```python
set()
```

para armazenar soluções distintas.

Exemplo:

```python
solucoes.add(tuple(x))
```

---

# Busca das 92 Soluções

Após encontrar uma solução:

1. Reiniciar algoritmo.
2. Buscar nova solução.
3. Adicionar ao conjunto.
4. Ignorar duplicatas.

Continuar enquanto:

```python
len(solucoes) < 92
```

---

# Estatísticas Necessárias

Registrar:

* solução encontrada
* valor da função objetivo
* número de iterações
* temperatura final
* tempo de execução

---

# Experimento Principal

Executar:

```python
R = 100
```

rodadas independentes.

Calcular:

* média de iterações
* média de tempo
* desvio padrão
* melhor resultado
* pior resultado

---

# Experimento das 92 Soluções

Registrar:

* quantidade de reinicializações
* tempo total
* iterações totais
* memória utilizada
* soluções encontradas

---

# Resultados Esperados

Tabela exemplo:

| Temperatura | Alpha | Média Iterações | Taxa Sucesso |
| ----------- | ----- | --------------- | ------------ |
| 10          | 0.99  | ...             | ...          |
| 50          | 0.99  | ...             | ...          |
| 100         | 0.99  | ...             | ...          |

---

# Gráficos

Gerar:

## Convergência

```text
iteração × fitness
```

---

## Temperatura

```text
iteração × temperatura
```

---

## Distribuição

```text
histograma das iterações
```

---

# Observações Importantes

Não utilizar bibliotecas prontas de:

* Simulated Annealing
* Meta-Heurísticas

Toda a lógica deve ser implementada manualmente.

Bibliotecas permitidas:

* numpy
* matplotlib
* pandas
* math
* random
* time

---

# Critério de Sucesso

Uma solução é considerada ótima quando:

```python
f(x) = 28
```

ou equivalentemente:

```python
h(x) = 0
```

Nenhuma rainha ataca outra.

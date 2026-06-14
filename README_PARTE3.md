# AV3 - Parte 3: Algoritmo Genético para TSP 3D

A implementação está isolada em `src3/` e não modifica as Partes 1 e 2.

O arquivo fornecido contém uma origem no grupo 0 e quatro grupos de 40 pontos.
Por padrão, a execução utiliza o grupo 1. A origem não integra o cromossomo:
ela é adicionada antes do primeiro ponto e após o último no cálculo da rota.

## Execução completa

```powershell
python -m src3.main
```

O modo padrão executa 100 rodadas com elitismo 0 e outras 100 com elitismo 5,
usando:

- população 100;
- máximo de 500 gerações;
- crossover OX com taxa 90%;
- swap mutation com taxa 1%;
- torneio de tamanho 3;
- parada após 50 gerações sem melhoria.

Os CSVs, resumos e gráficos são gravados em `results3/`.

## Execução rápida

```powershell
python -m src3.main --runs 3 --population-size 20 --max-generations 20 --no-plots
```

Outro grupo pode ser selecionado:

```powershell
python -m src3.main --group 4
```

## Solução aceitável

Como o enunciado não informa a distância ótima, a implementação estima a
distância média de 1000 rotas aleatórias. Por padrão, uma solução é considerada
aceitável ao atingir no máximo 75% dessa média. O valor pode ser alterado:

```powershell
python -m src3.main --acceptable-ratio 0.70
```

## Testes

```powershell
python -m unittest discover -s tests3 -v
```

# AV3 - Parte 2: Simulated Annealing para as 8 Rainhas

A Parte 2 está isolada no pacote `src2/` e não altera a implementação da Parte
1 em `src/`.

## Execução completa

```powershell
python -m src2.main
```

Essa execução realiza:

- 100 rodadas independentes com `T0=100` e `alpha=0.99`;
- 100 rodadas para cada combinação de `T0` e `alpha`;
- reinicializações até encontrar as 92 soluções distintas;
- geração de CSV, JSON, Markdown e gráficos em `results2/`.

## Execuções separadas

```powershell
python -m src2.main --mode experiment
python -m src2.main --mode grid
python -m src2.main --mode solutions
```

## Validação rápida

```powershell
python -m src2.main --mode experiment --runs 5 --max-iterations 1000 --no-plots
python -m src2.main --mode solutions --target-solutions 5 --quiet
```

## Testes da Parte 2

```powershell
python -m unittest discover -s tests2 -v
```

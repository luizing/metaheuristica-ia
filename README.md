# AV3 - Parte 1: otimização contínua

Implementação manual dos algoritmos Hill Climbing (HC), Local Random Search
(LRS) e Global Random Search (GRS) para as seis funções do enunciado.

## Execução completa

```powershell
python -m src.main
```

O modo padrão executa 100 rodadas independentes por combinação de algoritmo e
função, com 1000 iterações e parada após 100 iterações sem melhoria. Antes dos
experimentos finais, ele testa os valores pedidos de `epsilon` e `sigma`.

Os arquivos são gravados em `results/`:

- `resultados.md`: tabelas consolidadas;
- `parameter_studies.csv`: estudo de `epsilon` e `sigma`;
- `f1/` até `f6/`: resultados brutos, resumo em CSV/JSON e gráficos.

## Execução rápida

Útil para validar a instalação sem rodar o experimento completo:

```powershell
python -m src.main --runs 3 --study-runs 2 --max-iterations 50 --no-plots
```

É possível executar apenas algumas funções:

```powershell
python -m src.main --problems f1 f3
python -m src.experiments.experiment_f2
```

Para usar valores fixos, sem o estudo de hiperparâmetros:

```powershell
python -m src.main --no-study --epsilon 0.1 --sigma 0.1
```

## Testes

```powershell
python -m unittest discover -v
```

## Fórmulas recuperadas do PDF

O Markdown fornecido não continha as expressões de F5 e F6. A implementação usa
as fórmulas presentes no PDF original:

```text
F5 = x1*cos(x1)/20 + 2*exp(-x1²-(x2-1)²) + 0.01*x1*x2
F6 = x1*sin(4*pi*x1) - x2*sin(4*pi*x2+pi) + 1
```

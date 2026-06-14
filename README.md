# Meta-heuristicas e Inteligencia Artificial

Projeto dividido em tres partes:

1. `src`: otimizacao continua com Hill Climbing, Local Random Search e Global Random Search.
2. `src2`: problema das 8 rainhas resolvido com Tempera Simulada.
3. `src3`: Caixeiro Viajante 3D resolvido com Algoritmo Genetico.

## Requisitos

- Python 3.10 ou superior
- Dependencias listadas em `requirements.txt`

Na raiz do projeto, crie e ative um ambiente virtual e instale as dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No Linux ou macOS, a ativacao do ambiente virtual e feita com:

```bash
source .venv/bin/activate
```

Os comandos abaixo devem ser executados a partir da raiz do projeto.

## Parte 1: otimizacao continua (`src`)

Compara tres algoritmos em seis funcoes matematicas:

- `HC`: Hill Climbing, com perturbacao uniforme controlada por `epsilon`.
- `LRS`: Local Random Search, com perturbacao normal controlada por `sigma`.
- `GRS`: Global Random Search, com candidatos aleatorios em todo o dominio.

Executar todas as funcoes com os parametros padrao:

```powershell
python -m src.main
```

Executar somente algumas funcoes:

```powershell
python -m src.main --problems f1 f3 f4
```

Executar uma funcao pelo modulo individual:

```powershell
python -m src.experiments.experiment_f1
```

Existem modulos equivalentes de `experiment_f1` ate `experiment_f6`.

Exemplo de execucao reduzida:

```powershell
python -m src.main --runs 20 --study-runs 5 --max-iterations 500
```

Opcoes importantes:

- `--problems`: funcoes que serao executadas, entre `f1` e `f6`.
- `--runs`: quantidade de execucoes independentes por algoritmo.
- `--study-runs`: execucoes usadas no estudo de `epsilon` e `sigma`.
- `--max-iterations`: limite de iteracoes.
- `--stall-limit`: parada apos iteracoes consecutivas sem melhora.
- `--epsilon`: passo do Hill Climbing quando o estudo e desativado.
- `--sigma`: desvio da busca local quando o estudo e desativado.
- `--no-study`: desativa o estudo automatico de parametros.
- `--no-plots`: nao gera graficos.
- `--output`: diretorio de saida, com padrao `results`.
- `--seed`: semente aleatoria para reproducibilidade.

As saidas incluem CSV e JSON com resultados, relatorio Markdown e graficos de convergencia, distribuicao e histogramas.

### Modulos de `src`

- `main.py`: interface de linha de comando e coordenacao dos experimentos.
- `functions.py`: definicao das funcoes F1 a F6 e de seus dominios.
- `algorithms/`: implementacoes de HC, LRS e GRS.
- `core/problem.py`: representacao e avaliacao de um problema de otimizacao.
- `core/result.py`: estrutura que armazena o resultado de uma busca.
- `core/experiment.py`: execucoes repetidas, estatisticas e exportacao.
- `core/parameter_study.py`: selecao experimental de `epsilon` e `sigma`.
- `experiments/`: atalhos para executar separadamente cada funcao.
- `visualization.py`: geracao dos graficos da Parte 1.

## Parte 2: 8 rainhas (`src2`)

Utiliza Tempera Simulada para encontrar configuracoes sem conflitos e, por meio de reinicializacoes, coletar as 92 configuracoes conhecidas.

Executar o fluxo completo:

```powershell
python -m src2.main
```

Tambem pode ser usado o modulo de experimento:

```powershell
python -m src2.experiments.run_queens
```

Executar somente uma etapa:

```powershell
python -m src2.main --mode experiment
python -m src2.main --mode grid
python -m src2.main --mode solutions
```

Exemplo de execucao reduzida:

```powershell
python -m src2.main --mode experiment --runs 20 --max-iterations 5000
```

Opcoes importantes:

- `--mode`: `all`, `experiment`, `grid` ou `solutions`.
- `--runs`: execucoes do experimento principal.
- `--grid-runs`: execucoes para cada combinacao de parametros.
- `--temperature`: temperatura inicial.
- `--alpha`: fator do resfriamento geometrico.
- `--max-iterations`: limite por reinicializacao.
- `--target-solutions`: quantidade de solucoes distintas a coletar.
- `--max-restarts`: limite de reinicializacoes da coleta.
- `--no-plots`: nao gera graficos.
- `--quiet`: oculta o progresso da coleta.
- `--output`: diretorio de saida, com padrao `results2`.
- `--seed`: semente aleatoria.

As saidas incluem execucoes individuais, resumo estatistico, grade de parametros, lista das solucoes e graficos de convergencia, temperatura e iteracoes.

### Modulos de `src2`

- `main.py`: interface de linha de comando e coordenacao das etapas.
- `annealing/simulated_annealing.py`: implementacao da Tempera Simulada.
- `annealing/cooling.py`: resfriamento geometrico da temperatura.
- `queens/board.py`: representacao e validacao do tabuleiro.
- `queens/objective.py`: contagem de conflitos e calculo do fitness.
- `queens/neighborhood.py`: geracao de configuracoes vizinhas.
- `queens/solution_repository.py`: armazenamento de solucoes sem duplicatas.
- `experiments/queens_experiment.py`: experimentos, grade de parametros e coleta.
- `experiments/export.py`: exportacao para CSV, JSON e Markdown.
- `visualization.py`: graficos da Parte 2.

## Parte 3: Caixeiro Viajante 3D (`src3`)

Aplica um Algoritmo Genetico para minimizar a distancia de uma rota 3D que parte da origem, visita todos os pontos e retorna a origem. O experimento compara populacoes com elite 0 e elite 5.

A Parte 3 requer um arquivo CSV de coordenadas. Por padrao, o programa procura `CaixeiroGruposGA.csv` na raiz. Caso o arquivo esteja em outro local, informe-o com `--csv`.

Executar com o CSV padrao:

```powershell
python -m src3.main
```

Executar informando o arquivo e o grupo:

```powershell
python -m src3.main --csv caminho\pontos.csv --group 1
```

Tambem pode ser usado o modulo de experimento:

```powershell
python -m src3.experiments.run_tsp --csv caminho\pontos.csv
```

Formato aceito:

- CSV com colunas `x,y,z`; a origem sera `(0,0,0)`.
- CSV com colunas `x,y,z,grupo`; linhas do grupo 0 podem definir a origem.
- O conjunto selecionado deve conter entre 31 e 59 pontos.

Exemplo de execucao reduzida:

```powershell
python -m src3.main --csv caminho\pontos.csv --runs 20 --max-generations 200
```

Opcoes importantes:

- `--csv`: caminho do arquivo de coordenadas.
- `--group`: grupo de pontos, entre 1 e 4.
- `--runs`: quantidade de execucoes independentes.
- `--population-size`: tamanho da populacao.
- `--max-generations`: limite de geracoes.
- `--crossover-rate`: probabilidade de crossover.
- `--mutation-rate`: probabilidade de mutacao.
- `--tournament-size`: quantidade de competidores na selecao.
- `--stall-limit`: parada apos geracoes consecutivas sem melhora.
- `--acceptable-ratio`: limite aceitavel relativo a media aleatoria.
- `--baseline-samples`: rotas usadas na referencia aleatoria.
- `--no-plots`: nao gera graficos.
- `--output`: diretorio de saida, com padrao `results3`.
- `--seed`: semente aleatoria.

As saidas incluem resultados por execucao, resumos com e sem elitismo, historico da melhor execucao e graficos de convergencia, distribuicao e melhor rota 3D.

### Modulos de `src3`

- `main.py`: interface de linha de comando e comparacao de elitismo.
- `ga/genetic_algorithm.py`: configuracao e ciclo do Algoritmo Genetico.
- `tsp/point.py`: leitura e validacao das coordenadas.
- `tsp/route.py`: validacao dos cromossomos que representam rotas.
- `tsp/fitness.py`: matriz de distancias, distancia da rota e fitness.
- `tsp/population.py`: criacao da populacao inicial.
- `tsp/selection.py`: selecao por torneio.
- `tsp/crossover.py`: crossover de ordem, apropriado para permutacoes.
- `tsp/mutation.py`: mutacao por troca de duas posicoes.
- `experiments/tsp_experiment.py`: repeticoes, baseline e comparacao de elites.
- `experiments/export.py`: exportacao para CSV, JSON e Markdown.
- `visualization.py`: graficos e visualizacao da rota 3D.

## Ajuda dos comandos

Todos os programas principais oferecem a lista completa de opcoes:

```powershell
python -m src.main --help
python -m src2.main --help
python -m src3.main --help
```

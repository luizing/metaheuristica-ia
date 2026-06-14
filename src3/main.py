from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src3.experiments.export import (
    export_comparison,
    export_experiment,
    export_generation_history,
    export_markdown,
)
from src3.experiments.tsp_experiment import compare_elitism
from src3.ga.genetic_algorithm import GAConfig
from src3.tsp.fitness import DistanceModel
from src3.tsp.point import load_points
from src3.visualization import create_comparison_plots, create_experiment_plots


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AV3 Parte 3 - Algoritmo Genético para TSP 3D."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("CaixeiroGruposGA.csv"),
    )
    parser.add_argument("--group", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--max-generations", type=int, default=500)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.01)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--stall-limit", type=int, default=50)
    parser.add_argument("--acceptable-ratio", type=float, default=0.75)
    parser.add_argument("--baseline-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results3"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def print_summary(elite_size: int, experiment) -> None:
    summary = experiment.summary
    print(
        f"Elite {elite_size}: melhor={summary.best_distance:.3f}, "
        f"média={summary.mean_distance:.3f}, "
        f"pior={summary.worst_distance:.3f}, "
        f"moda de gerações={summary.mode_generations}, "
        f"melhoria sobre rotas aleatórias="
        f"{summary.improvement_over_random_percent:.2f}%."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_points(args.csv, group=args.group)
    distance_model = DistanceModel(dataset)
    config = GAConfig(
        population_size=args.population_size,
        max_generations=args.max_generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
        elite_size=5,
        stall_limit=args.stall_limit,
    )
    comparisons = compare_elitism(
        distance_model,
        config,
        elite_sizes=(0, 5),
        runs=args.runs,
        seed=args.seed,
        baseline_samples=args.baseline_samples,
        acceptable_ratio=args.acceptable_ratio,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for elite_size, experiment in comparisons.items():
        directory = args.output / f"elite_{elite_size}"
        export_experiment(directory, experiment)
        export_generation_history(directory / "best_run_history.csv", experiment)
        if not args.no_plots:
            create_experiment_plots(directory, experiment, dataset)
        print_summary(elite_size, experiment)

    export_comparison(args.output / "elitism_comparison.csv", comparisons)
    export_markdown(args.output / "resultados.md", dataset, comparisons)
    if not args.no_plots:
        create_comparison_plots(args.output, comparisons)
    print(f"Resultados gravados em: {args.output.resolve()}")


if __name__ == "__main__":
    main()

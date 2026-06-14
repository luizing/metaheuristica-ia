from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src2.experiments.export import (
    export_collection,
    export_independent_results,
    export_markdown_report,
    export_parameter_grid,
)
from src2.experiments.queens_experiment import (
    ALPHA_VALUES,
    TEMPERATURE_VALUES,
    CollectionResult,
    ExperimentSummary,
    ParameterSummary,
    collect_distinct_solutions,
    run_independent_experiment,
    run_parameter_grid,
)
from src2.visualization import create_experiment_plots


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AV3 Parte 2 - Simulated Annealing para as 8 Rainhas."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "experiment", "grid", "solutions"),
        default="all",
        help="Etapa a executar.",
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--grid-runs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--target-solutions", type=int, default=92)
    parser.add_argument("--max-restarts", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results2"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def print_experiment_summary(summary: ExperimentSummary) -> None:
    print(
        f"Experimento: {summary.successes}/{summary.runs} sucessos "
        f"({summary.success_rate:.2%}), média de "
        f"{summary.mean_iterations:.2f} iterações e "
        f"{summary.mean_seconds:.6f} s."
    )


def print_parameter_grid(parameters: list[ParameterSummary]) -> None:
    print("\nTemperatura | Alpha | Média iterações | Taxa de sucesso")
    for parameter in parameters:
        print(
            f"{parameter.initial_temperature:11g} | "
            f"{parameter.alpha:5g} | "
            f"{parameter.summary.mean_iterations:16.2f} | "
            f"{parameter.summary.success_rate:14.2%}"
        )


def print_collection(collection: CollectionResult) -> None:
    print(
        f"Coleta: {len(collection.repository)} soluções distintas em "
        f"{collection.restarts} reinicializações, "
        f"{collection.total_iterations} iterações e "
        f"{collection.elapsed_seconds:.3f} s."
    )
    print(
        f"Pico de memória: {collection.peak_memory_bytes / (1024**2):.3f} MB; "
        f"duplicatas: {collection.duplicate_solutions}; "
        f"falhas: {collection.failed_restarts}."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    experiment_summary = None
    parameter_summaries = None
    collection = None

    if args.mode in {"all", "experiment"}:
        results, experiment_summary = run_independent_experiment(
            runs=args.runs,
            initial_temperature=args.temperature,
            alpha=args.alpha,
            max_iterations=args.max_iterations,
            seed=args.seed,
            record_history=not args.no_plots,
        )
        experiment_directory = args.output / "experiment"
        export_independent_results(
            experiment_directory,
            results,
            experiment_summary,
        )
        if not args.no_plots:
            create_experiment_plots(experiment_directory, results)
        print_experiment_summary(experiment_summary)

    if args.mode in {"all", "grid"}:
        parameter_summaries = run_parameter_grid(
            runs=args.grid_runs,
            temperatures=TEMPERATURE_VALUES,
            alphas=ALPHA_VALUES,
            max_iterations=args.max_iterations,
            seed=args.seed + 1,
        )
        export_parameter_grid(
            args.output / "parameter_grid.csv",
            parameter_summaries,
        )
        print_parameter_grid(parameter_summaries)

    if args.mode in {"all", "solutions"}:
        progress = None
        if not args.quiet:
            progress = lambda count, restart: print(
                f"Solução {count}/{args.target_solutions} "
                f"encontrada na reinicialização {restart}."
            )
        collection = collect_distinct_solutions(
            target_count=args.target_solutions,
            initial_temperature=args.temperature,
            alpha=args.alpha,
            max_iterations=args.max_iterations,
            max_restarts=args.max_restarts,
            seed=args.seed + 2,
            progress=progress,
        )
        export_collection(args.output / "solutions", collection)
        print_collection(collection)

    export_markdown_report(
        args.output / "resultados.md",
        experiment_summary,
        parameter_summaries,
        collection,
    )
    print(f"Resultados gravados em: {args.output.resolve()}")


if __name__ == "__main__":
    main()

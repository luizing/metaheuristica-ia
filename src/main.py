from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.core.experiment import (
    ExperimentConfig,
    export_markdown,
    export_raw_results,
    export_summaries,
    markdown_table,
    run_problem,
)
from src.core.parameter_study import (
    EPSILON_VALUES,
    SIGMA_VALUES,
    ParameterResult,
    study_parameter,
)
from src.functions import PROBLEMS, get_problem
from src.visualization import create_plots


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AV3 Parte 1 - otimização contínua com meta-heurísticas."
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=[f"f{index}" for index in range(1, 7)],
        help="Problemas a executar: f1 f2 ... f6.",
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--study-runs", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--stall-limit", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--mode-decimals", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-study", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def write_parameter_study(
    path: Path,
    problem_name: str,
    algorithm: str,
    selected: float,
    results: list[ParameterResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if write_header:
            writer.writerow(
                [
                    "problem",
                    "algorithm",
                    "parameter",
                    "best",
                    "mean",
                    "reached_reference",
                    "selected",
                ]
            )
        for result in results:
            writer.writerow(
                [
                    problem_name,
                    algorithm,
                    result.value,
                    result.best,
                    result.mean,
                    result.reached_reference,
                    result.value == selected,
                ]
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    problems = [get_problem(identifier) for identifier in args.problems]
    base_config = ExperimentConfig(
        runs=args.runs,
        max_iterations=args.max_iterations,
        stall_limit=args.stall_limit,
        epsilon=args.epsilon,
        sigma=args.sigma,
        mode_decimals=args.mode_decimals,
        seed=args.seed,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    study_path = output / "parameter_studies.csv"
    if study_path.exists() and not args.no_study:
        study_path.unlink()

    tables = []
    for problem_index, problem in enumerate(problems):
        config = replace(base_config, seed=base_config.seed + problem_index * 100_000)

        if not args.no_study:
            study_config = replace(config, runs=args.study_runs)
            epsilon, epsilon_results = study_parameter(
                problem,
                "HC",
                EPSILON_VALUES,
                study_config,
            )
            sigma, sigma_results = study_parameter(
                problem,
                "LRS",
                SIGMA_VALUES,
                study_config,
            )
            config = replace(config, epsilon=epsilon, sigma=sigma)
            write_parameter_study(
                study_path, problem.name, "HC", epsilon, epsilon_results
            )
            write_parameter_study(
                study_path, problem.name, "LRS", sigma, sigma_results
            )
            print(
                f"{problem.name}: epsilon selecionado={epsilon:g}; "
                f"sigma selecionado={sigma:g}"
            )

        all_results, summaries = run_problem(problem, config)
        problem_directory = output / problem.name.split(" - ", 1)[0].lower()
        export_raw_results(
            problem_directory / "runs.csv",
            problem,
            all_results,
        )
        export_summaries(
            problem_directory / "summary.csv",
            problem_directory / "summary.json",
            problem,
            summaries,
        )
        if not args.no_plots:
            create_plots(problem_directory, problem, all_results)

        table = markdown_table(problem, summaries)
        tables.append(table)
        print(table)
        print()

    export_markdown(output / "resultados.md", tables)
    print(f"Resultados gravados em: {output.resolve()}")


if __name__ == "__main__":
    main()

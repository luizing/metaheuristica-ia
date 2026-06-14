from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import multimode
from typing import Callable, Iterable

import numpy as np

from src.algorithms import (
    global_random_search,
    hill_climbing,
    local_random_search,
)
from src.core.problem import Problem
from src.core.result import SearchResult

Algorithm = Callable[..., SearchResult]


@dataclass(frozen=True)
class ExperimentConfig:
    runs: int = 100
    max_iterations: int = 1000
    stall_limit: int = 100
    epsilon: float = 0.1
    sigma: float = 0.1
    mode_decimals: int = 6
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("runs must be positive.")


@dataclass(frozen=True)
class Summary:
    algorithm: str
    best: float
    worst: float
    mean: float
    standard_deviation: float
    mode: float
    best_x: tuple[float, ...]
    mean_iterations: float
    mean_evaluations: float


ALGORITHMS: dict[str, Algorithm] = {
    "HC": hill_climbing,
    "LRS": local_random_search,
    "GRS": global_random_search,
}


def run_algorithm(
    problem: Problem,
    algorithm_name: str,
    config: ExperimentConfig,
    *,
    seed_sequence: np.random.SeedSequence | None = None,
) -> list[SearchResult]:
    name = algorithm_name.upper()
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm: {algorithm_name}")

    root_seed = seed_sequence or np.random.SeedSequence(config.seed)
    run_seeds = root_seed.spawn(config.runs)
    common = {
        "max_iterations": config.max_iterations,
        "stall_limit": config.stall_limit,
    }
    specific = (
        {"epsilon": config.epsilon}
        if name == "HC"
        else {"sigma": config.sigma}
        if name == "LRS"
        else {}
    )

    return [
        ALGORITHMS[name](
            problem,
            np.random.default_rng(run_seed),
            **common,
            **specific,
        )
        for run_seed in run_seeds
    ]


def summarize(
    problem: Problem,
    results: list[SearchResult],
    *,
    mode_decimals: int = 6,
) -> Summary:
    if not results:
        raise ValueError("Cannot summarize an empty result list.")

    values = np.array([result.best_value for result in results])
    best_index = problem.best_index(values)
    worst_index = problem.worst_index(values)
    rounded = np.round(values, decimals=mode_decimals)
    modes = multimode(rounded.tolist())
    mode = max(modes) if problem.maximize else min(modes)

    return Summary(
        algorithm=results[0].algorithm,
        best=float(values[best_index]),
        worst=float(values[worst_index]),
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        mode=float(mode),
        best_x=tuple(float(value) for value in results[best_index].x_best),
        mean_iterations=float(np.mean([result.iterations for result in results])),
        mean_evaluations=float(np.mean([result.evaluations for result in results])),
    )


def run_problem(
    problem: Problem,
    config: ExperimentConfig,
) -> tuple[dict[str, list[SearchResult]], dict[str, Summary]]:
    algorithm_seeds = np.random.SeedSequence(config.seed).spawn(len(ALGORITHMS))
    all_results = {
        name: run_algorithm(
            problem,
            name,
            config,
            seed_sequence=algorithm_seeds[index],
        )
        for index, name in enumerate(ALGORITHMS)
    }
    summaries = {
        name: summarize(
            problem,
            results,
            mode_decimals=config.mode_decimals,
        )
        for name, results in all_results.items()
    }
    return all_results, summaries


def mean_convergence(results: list[SearchResult]) -> np.ndarray:
    max_length = max(len(result.history) for result in results)
    histories = np.empty((len(results), max_length))
    for index, result in enumerate(results):
        histories[index, : len(result.history)] = result.history
        histories[index, len(result.history) :] = result.history[-1]
    return np.mean(histories, axis=0)


def export_raw_results(
    path: Path,
    problem: Problem,
    all_results: dict[str, list[SearchResult]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "problem",
                "algorithm",
                "run",
                "best_value",
                "iterations",
                "evaluations",
                *[f"x{i}" for i in range(1, problem.dimension + 1)],
            ]
        )
        for algorithm, results in all_results.items():
            for run, result in enumerate(results, start=1):
                writer.writerow(
                    [
                        problem.name,
                        algorithm,
                        run,
                        result.best_value,
                        result.iterations,
                        result.evaluations,
                        *result.x_best,
                    ]
                )


def export_summaries(
    csv_path: Path,
    json_path: Path,
    problem: Problem,
    summaries: dict[str, Summary],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "problem": problem.name,
            **asdict(summary),
            "best_x": json.dumps(summary.best_x),
        }
        for summary in summaries.values()
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def markdown_table(problem: Problem, summaries: dict[str, Summary]) -> str:
    lines = [
        f"## {problem.name}",
        "",
        "| Algoritmo | Moda | Média | Desvio padrão | Melhor | Pior | x do melhor |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ALGORITHMS:
        summary = summaries[name]
        best_x = ", ".join(f"{value:.6f}" for value in summary.best_x)
        lines.append(
            f"| {name} | {summary.mode:.8g} | {summary.mean:.8g} | "
            f"{summary.standard_deviation:.8g} | {summary.best:.8g} | "
            f"{summary.worst:.8g} | ({best_x}) |"
        )
    return "\n".join(lines)


def export_markdown(path: Path, tables: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Resultados da otimização contínua\n\n" + "\n\n".join(tables) + "\n",
        encoding="utf-8",
    )

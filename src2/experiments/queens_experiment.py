from __future__ import annotations

import tracemalloc
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from src2.annealing.simulated_annealing import (
    AnnealingResult,
    simulated_annealing,
)
from src2.queens.solution_repository import SolutionRepository

TEMPERATURE_VALUES = (10.0, 50.0, 100.0, 500.0)
ALPHA_VALUES = (0.95, 0.97, 0.99, 0.995)


@dataclass(frozen=True)
class ExperimentSummary:
    runs: int
    successes: int
    success_rate: float
    mean_iterations: float
    standard_deviation_iterations: float
    mean_seconds: float
    standard_deviation_seconds: float
    best_fitness: int
    worst_fitness: int
    best_iterations: int
    worst_iterations: int


@dataclass(frozen=True)
class ParameterSummary:
    initial_temperature: float
    alpha: float
    summary: ExperimentSummary


@dataclass(frozen=True)
class SolutionDiscovery:
    solution_number: int
    restart: int
    solution: tuple[int, ...]
    iterations: int
    elapsed_seconds: float
    final_temperature: float


@dataclass(frozen=True)
class CollectionResult:
    repository: SolutionRepository
    restarts: int
    successful_restarts: int
    duplicate_solutions: int
    failed_restarts: int
    total_iterations: int
    elapsed_seconds: float
    peak_memory_bytes: int
    discoveries: tuple[SolutionDiscovery, ...]


def summarize_results(results: list[AnnealingResult]) -> ExperimentSummary:
    if not results:
        raise ValueError("At least one result is required.")

    iterations = np.asarray([result.iterations for result in results], dtype=float)
    elapsed = np.asarray(
        [result.elapsed_seconds for result in results],
        dtype=float,
    )
    fitness_values = [result.fitness for result in results]
    successes = sum(result.success for result in results)

    return ExperimentSummary(
        runs=len(results),
        successes=successes,
        success_rate=successes / len(results),
        mean_iterations=float(np.mean(iterations)),
        standard_deviation_iterations=(
            float(np.std(iterations, ddof=1)) if len(results) > 1 else 0.0
        ),
        mean_seconds=float(np.mean(elapsed)),
        standard_deviation_seconds=(
            float(np.std(elapsed, ddof=1)) if len(results) > 1 else 0.0
        ),
        best_fitness=max(fitness_values),
        worst_fitness=min(fitness_values),
        best_iterations=int(np.min(iterations)),
        worst_iterations=int(np.max(iterations)),
    )


def run_independent_experiment(
    *,
    runs: int = 100,
    initial_temperature: float = 100.0,
    alpha: float = 0.99,
    max_iterations: int = 10_000,
    seed: int = 2026,
    record_history: bool = True,
) -> tuple[list[AnnealingResult], ExperimentSummary]:
    if runs <= 0:
        raise ValueError("runs must be positive.")

    run_seeds = np.random.SeedSequence(seed).spawn(runs)
    results = [
        simulated_annealing(
            np.random.default_rng(run_seed),
            initial_temperature=initial_temperature,
            alpha=alpha,
            max_iterations=max_iterations,
            record_history=record_history,
        )
        for run_seed in run_seeds
    ]
    return results, summarize_results(results)


def run_parameter_grid(
    *,
    runs: int = 100,
    temperatures: tuple[float, ...] = TEMPERATURE_VALUES,
    alphas: tuple[float, ...] = ALPHA_VALUES,
    max_iterations: int = 10_000,
    seed: int = 2026,
) -> list[ParameterSummary]:
    combination_seeds = np.random.SeedSequence(seed).spawn(
        len(temperatures) * len(alphas)
    )
    summaries = []
    seed_index = 0

    for temperature in temperatures:
        for alpha in alphas:
            results, summary = run_independent_experiment(
                runs=runs,
                initial_temperature=temperature,
                alpha=alpha,
                max_iterations=max_iterations,
                seed=int(combination_seeds[seed_index].generate_state(1)[0]),
                record_history=False,
            )
            del results
            summaries.append(
                ParameterSummary(
                    initial_temperature=temperature,
                    alpha=alpha,
                    summary=summary,
                )
            )
            seed_index += 1

    return summaries


def collect_distinct_solutions(
    *,
    target_count: int = 92,
    initial_temperature: float = 100.0,
    alpha: float = 0.99,
    max_iterations: int = 10_000,
    max_restarts: int = 100_000,
    seed: int = 2026,
    progress: Callable[[int, int], None] | None = None,
) -> CollectionResult:
    if not 1 <= target_count <= 92:
        raise ValueError("target_count must be between 1 and 92.")
    if max_restarts <= 0:
        raise ValueError("max_restarts must be positive.")

    repository = SolutionRepository()
    discoveries: list[SolutionDiscovery] = []
    successful_restarts = 0
    duplicate_solutions = 0
    failed_restarts = 0
    total_iterations = 0
    seed_generator = np.random.default_rng(seed)
    start = perf_counter()
    tracemalloc.start()

    try:
        for restart in range(1, max_restarts + 1):
            restart_seed = int(
                seed_generator.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            )
            result = simulated_annealing(
                np.random.default_rng(restart_seed),
                initial_temperature=initial_temperature,
                alpha=alpha,
                max_iterations=max_iterations,
                record_history=False,
            )
            total_iterations += result.iterations

            if result.success:
                successful_restarts += 1
                if repository.add(result.state):
                    discoveries.append(
                        SolutionDiscovery(
                            solution_number=len(repository),
                            restart=restart,
                            solution=tuple(int(row) for row in result.state),
                            iterations=result.iterations,
                            elapsed_seconds=result.elapsed_seconds,
                            final_temperature=result.final_temperature,
                        )
                    )
                    if progress is not None:
                        progress(len(repository), restart)
                else:
                    duplicate_solutions += 1
            else:
                failed_restarts += 1

            if len(repository) == target_count:
                break
        else:
            raise RuntimeError(
                f"Only {len(repository)} distinct solutions were found after "
                f"{max_restarts} restarts."
            )
    finally:
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return CollectionResult(
        repository=repository,
        restarts=restart,
        successful_restarts=successful_restarts,
        duplicate_solutions=duplicate_solutions,
        failed_restarts=failed_restarts,
        total_iterations=total_iterations,
        elapsed_seconds=perf_counter() - start,
        peak_memory_bytes=peak_memory,
        discoveries=tuple(discoveries),
    )

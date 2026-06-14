from dataclasses import dataclass
from statistics import multimode

import numpy as np

from src3.ga.genetic_algorithm import GAConfig, GAResult, GeneticAlgorithm
from src3.tsp.fitness import DistanceModel


@dataclass(frozen=True)
class RandomBaseline:
    samples: int
    best_distance: float
    mean_distance: float
    worst_distance: float


@dataclass(frozen=True)
class ExperimentSummary:
    runs: int
    elite_size: int
    best_distance: float
    worst_distance: float
    mean_distance: float
    standard_deviation_distance: float
    mode_generations: int
    mean_generations: float
    mean_seconds: float
    acceptable_distance: float
    acceptable_successes: int
    acceptable_success_rate: float
    mean_generation_to_acceptable: float | None
    random_mean_distance: float
    improvement_over_random_percent: float


@dataclass(frozen=True)
class ExperimentResult:
    results: tuple[GAResult, ...]
    summary: ExperimentSummary
    baseline: RandomBaseline


def estimate_random_baseline(
    distance_model: DistanceModel,
    *,
    samples: int = 1_000,
    seed: int = 2026,
) -> RandomBaseline:
    if samples <= 0:
        raise ValueError("samples must be positive.")
    rng = np.random.default_rng(seed)
    population = np.asarray(
        [rng.permutation(distance_model.point_count) for _ in range(samples)],
        dtype=np.int32,
    )
    distances = distance_model.population_distances(population)
    return RandomBaseline(
        samples=samples,
        best_distance=float(np.min(distances)),
        mean_distance=float(np.mean(distances)),
        worst_distance=float(np.max(distances)),
    )


def summarize_experiment(
    results: list[GAResult],
    config: GAConfig,
    baseline: RandomBaseline,
    acceptable_distance: float,
) -> ExperimentSummary:
    if not results:
        raise ValueError("At least one result is required.")

    distances = np.asarray([result.best.distance for result in results])
    generations = [result.generations for result in results]
    acceptable_generations = [
        result.generations_to_acceptable
        for result in results
        if result.generations_to_acceptable is not None
    ]
    generations_for_mode = acceptable_generations or generations
    generation_modes = multimode(generations_for_mode)
    improvement = (
        (baseline.mean_distance - float(np.mean(distances)))
        / baseline.mean_distance
        * 100.0
    )

    return ExperimentSummary(
        runs=len(results),
        elite_size=config.elite_size,
        best_distance=float(np.min(distances)),
        worst_distance=float(np.max(distances)),
        mean_distance=float(np.mean(distances)),
        standard_deviation_distance=(
            float(np.std(distances, ddof=1)) if len(results) > 1 else 0.0
        ),
        mode_generations=min(generation_modes),
        mean_generations=float(np.mean(generations)),
        mean_seconds=float(np.mean([result.elapsed_seconds for result in results])),
        acceptable_distance=acceptable_distance,
        acceptable_successes=len(acceptable_generations),
        acceptable_success_rate=len(acceptable_generations) / len(results),
        mean_generation_to_acceptable=(
            float(np.mean(acceptable_generations))
            if acceptable_generations
            else None
        ),
        random_mean_distance=baseline.mean_distance,
        improvement_over_random_percent=improvement,
    )


def run_experiment(
    distance_model: DistanceModel,
    config: GAConfig,
    *,
    runs: int = 100,
    seed: int = 2026,
    baseline_samples: int = 1_000,
    acceptable_ratio: float = 0.75,
    baseline: RandomBaseline | None = None,
) -> ExperimentResult:
    if runs <= 0:
        raise ValueError("runs must be positive.")
    if not 0.0 < acceptable_ratio < 1.0:
        raise ValueError("acceptable_ratio must be between 0 and 1.")

    baseline = baseline or estimate_random_baseline(
        distance_model,
        samples=baseline_samples,
        seed=seed + 1,
    )
    acceptable_distance = baseline.mean_distance * acceptable_ratio
    run_seeds = np.random.SeedSequence(seed).spawn(runs)
    results = [
        GeneticAlgorithm(
            distance_model,
            config,
            np.random.default_rng(run_seed),
        ).run(acceptable_distance=acceptable_distance)
        for run_seed in run_seeds
    ]
    return ExperimentResult(
        results=tuple(results),
        summary=summarize_experiment(
            results,
            config,
            baseline,
            acceptable_distance,
        ),
        baseline=baseline,
    )


def compare_elitism(
    distance_model: DistanceModel,
    base_config: GAConfig,
    *,
    elite_sizes: tuple[int, ...] = (0, 5),
    runs: int = 100,
    seed: int = 2026,
    baseline_samples: int = 1_000,
    acceptable_ratio: float = 0.75,
) -> dict[int, ExperimentResult]:
    baseline = estimate_random_baseline(
        distance_model,
        samples=baseline_samples,
        seed=seed + 1,
    )
    comparisons = {}
    for elite_size in elite_sizes:
        config = GAConfig(
            population_size=base_config.population_size,
            max_generations=base_config.max_generations,
            crossover_rate=base_config.crossover_rate,
            mutation_rate=base_config.mutation_rate,
            tournament_size=base_config.tournament_size,
            elite_size=elite_size,
            stall_limit=base_config.stall_limit,
        )
        comparisons[elite_size] = run_experiment(
            distance_model,
            config,
            runs=runs,
            seed=seed,
            baseline_samples=baseline_samples,
            acceptable_ratio=acceptable_ratio,
            baseline=baseline,
        )
    return comparisons

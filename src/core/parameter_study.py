from dataclasses import dataclass

import numpy as np

from src.core.experiment import ExperimentConfig, run_algorithm, summarize
from src.core.problem import Problem

EPSILON_VALUES = (0.5, 0.2, 0.1, 0.05, 0.01)
SIGMA_VALUES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)


@dataclass(frozen=True)
class ParameterResult:
    value: float
    best: float
    mean: float
    reached_reference: bool


def estimate_reference_value(problem: Problem) -> float:
    if problem.known_optimum is not None:
        return problem.evaluate(np.array(problem.known_optimum))

    lower = problem.lower_bounds
    upper = problem.upper_bounds
    center = (lower + upper) / 2.0
    radius = (upper - lower) / 2.0
    best_x = center
    best_value = problem.evaluate(best_x)

    for points in (201, 101, 101):
        axes = [
            np.linspace(
                max(lower[dim], best_x[dim] - radius[dim]),
                min(upper[dim], best_x[dim] + radius[dim]),
                points,
            )
            for dim in range(problem.dimension)
        ]
        for coordinates in np.array(np.meshgrid(*axes)).T.reshape(-1, problem.dimension):
            value = problem.evaluate(coordinates)
            if problem.is_better(value, best_value):
                best_x = coordinates.copy()
                best_value = value
        radius /= 10.0

    return best_value


def study_parameter(
    problem: Problem,
    algorithm: str,
    values: tuple[float, ...],
    config: ExperimentConfig,
    *,
    tolerance: float = 1e-3,
) -> tuple[float, list[ParameterResult]]:
    reference = estimate_reference_value(problem)
    study_results = []

    for index, value in enumerate(values):
        candidate_config = ExperimentConfig(
            runs=config.runs,
            max_iterations=config.max_iterations,
            stall_limit=config.stall_limit,
            epsilon=value if algorithm == "HC" else config.epsilon,
            sigma=value if algorithm == "LRS" else config.sigma,
            mode_decimals=config.mode_decimals,
            seed=config.seed + 10_000 * (index + 1),
        )
        results = run_algorithm(problem, algorithm, candidate_config)
        summary = summarize(
            problem,
            results,
            mode_decimals=config.mode_decimals,
        )
        error = abs(summary.best - reference)
        threshold = tolerance * max(1.0, abs(reference))
        study_results.append(
            ParameterResult(
                value=value,
                best=summary.best,
                mean=summary.mean,
                reached_reference=error <= threshold,
            )
        )

    successful = [result.value for result in study_results if result.reached_reference]
    if successful:
        return min(successful), study_results

    means = np.array([result.mean for result in study_results])
    fallback_index = problem.best_index(means)
    return study_results[fallback_index].value, study_results

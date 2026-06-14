import numpy as np

from src.core.problem import Problem


def validate_search_parameters(max_iterations: int, stall_limit: int) -> None:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if stall_limit <= 0:
        raise ValueError("stall_limit must be positive.")


def uniform_candidate(problem: Problem, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(problem.lower_bounds, problem.upper_bounds)

import numpy as np

from src.algorithms.common import uniform_candidate, validate_search_parameters
from src.core.problem import Problem
from src.core.result import SearchResult


def global_random_search(
    problem: Problem,
    rng: np.random.Generator,
    *,
    max_iterations: int = 1000,
    stall_limit: int = 100,
) -> SearchResult:
    validate_search_parameters(max_iterations, stall_limit)

    x_best = uniform_candidate(problem, rng)
    best_value = problem.evaluate(x_best)
    history = [best_value]
    evaluations = 1
    stall = 0

    for iteration in range(1, max_iterations + 1):
        candidate = uniform_candidate(problem, rng)
        candidate_value = problem.evaluate(candidate)
        evaluations += 1

        if problem.is_better(candidate_value, best_value):
            x_best = candidate
            best_value = candidate_value
            stall = 0
        else:
            stall += 1

        history.append(best_value)
        if stall >= stall_limit:
            break

    return SearchResult(
        algorithm="GRS",
        x_best=x_best.copy(),
        best_value=best_value,
        iterations=iteration,
        evaluations=evaluations,
        history=np.asarray(history),
    )

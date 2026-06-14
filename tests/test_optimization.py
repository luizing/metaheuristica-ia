import unittest

import numpy as np

from src.algorithms import (
    global_random_search,
    hill_climbing,
    local_random_search,
)
from src.core.experiment import ExperimentConfig, run_algorithm, summarize
from src.functions import PROBLEMS, sphere


class ProblemTests(unittest.TestCase):
    def test_clip_respects_each_dimension(self) -> None:
        problem = PROBLEMS[1]
        clipped = problem.clip(np.array([-20.0, 30.0]))
        np.testing.assert_array_equal(clipped, np.array([-2.0, 5.0]))

    def test_all_functions_are_finite_at_bounds_and_center(self) -> None:
        for problem in PROBLEMS:
            for candidate in (
                problem.lower_bounds,
                problem.upper_bounds,
                (problem.lower_bounds + problem.upper_bounds) / 2.0,
            ):
                self.assertTrue(np.isfinite(problem.evaluate(candidate)))

    def test_known_minimum_of_sphere(self) -> None:
        self.assertEqual(sphere(np.zeros(2)), 0.0)


class AlgorithmTests(unittest.TestCase):
    def test_hill_climbing_starts_at_lower_bound(self) -> None:
        problem = PROBLEMS[0]
        result = hill_climbing(
            problem,
            np.random.default_rng(1),
            epsilon=1.0,
            max_iterations=5,
            stall_limit=5,
        )
        self.assertLessEqual(result.best_value, problem.evaluate(problem.lower_bounds))

    def test_all_histories_are_monotonic_for_minimization(self) -> None:
        problem = PROBLEMS[0]
        algorithms = (
            lambda rng: hill_climbing(
                problem, rng, epsilon=1.0, max_iterations=50, stall_limit=20
            ),
            lambda rng: local_random_search(
                problem, rng, sigma=1.0, max_iterations=50, stall_limit=20
            ),
            lambda rng: global_random_search(
                problem, rng, max_iterations=50, stall_limit=20
            ),
        )
        for index, algorithm in enumerate(algorithms):
            result = algorithm(np.random.default_rng(index))
            self.assertTrue(np.all(np.diff(result.history) <= 0.0))
            self.assertTrue(
                np.all(result.x_best >= problem.lower_bounds)
                and np.all(result.x_best <= problem.upper_bounds)
            )

    def test_reproducible_independent_runs(self) -> None:
        config = ExperimentConfig(
            runs=4,
            max_iterations=30,
            stall_limit=10,
            seed=123,
        )
        first = run_algorithm(PROBLEMS[0], "GRS", config)
        second = run_algorithm(PROBLEMS[0], "GRS", config)
        np.testing.assert_allclose(
            [result.best_value for result in first],
            [result.best_value for result in second],
        )

    def test_summary_uses_optimization_direction(self) -> None:
        config = ExperimentConfig(
            runs=5,
            max_iterations=20,
            stall_limit=10,
            seed=55,
        )
        results = run_algorithm(PROBLEMS[1], "GRS", config)
        summary = summarize(PROBLEMS[1], results)
        values = [result.best_value for result in results]
        self.assertEqual(summary.best, max(values))
        self.assertEqual(summary.worst, min(values))


if __name__ == "__main__":
    unittest.main()

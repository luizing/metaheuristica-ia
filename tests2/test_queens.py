import unittest

import numpy as np

from src2.annealing import GeometricCooling, simulated_annealing
from src2.queens.board import format_board
from src2.queens.neighborhood import perturb
from src2.queens.objective import attacking_pairs, fitness, is_solution
from src2.queens.solution_repository import SolutionRepository

VALID_SOLUTION = np.array([1, 5, 8, 6, 3, 7, 2, 4], dtype=np.int8)


class QueensTests(unittest.TestCase):
    def test_valid_solution_has_maximum_fitness(self) -> None:
        self.assertEqual(attacking_pairs(VALID_SOLUTION), 0)
        self.assertEqual(fitness(VALID_SOLUTION), 28)
        self.assertTrue(is_solution(VALID_SOLUTION))

    def test_all_queens_in_same_row_have_28_conflicts(self) -> None:
        state = np.ones(8, dtype=np.int8)
        self.assertEqual(attacking_pairs(state), 28)
        self.assertEqual(fitness(state), 0)

    def test_neighborhood_changes_exactly_one_column(self) -> None:
        candidate = perturb(VALID_SOLUTION, np.random.default_rng(10))
        self.assertEqual(np.count_nonzero(candidate != VALID_SOLUTION), 1)
        self.assertTrue(np.all((candidate >= 1) & (candidate <= 8)))

    def test_repository_ignores_duplicates(self) -> None:
        repository = SolutionRepository()
        self.assertTrue(repository.add(VALID_SOLUTION))
        self.assertFalse(repository.add(VALID_SOLUTION.copy()))
        self.assertEqual(len(repository), 1)

    def test_board_contains_eight_queens(self) -> None:
        self.assertEqual(format_board(VALID_SOLUTION).count("Q"), 8)


class AnnealingTests(unittest.TestCase):
    def test_geometric_cooling(self) -> None:
        cooling = GeometricCooling(alpha=0.99)
        self.assertAlmostEqual(cooling.cool(100.0), 99.0)

    def test_known_solution_stops_immediately(self) -> None:
        result = simulated_annealing(
            np.random.default_rng(1),
            initial_state=VALID_SOLUTION,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.fitness, 28)

    def test_best_fitness_history_never_decreases(self) -> None:
        result = simulated_annealing(
            np.random.default_rng(2),
            max_iterations=200,
        )
        self.assertTrue(np.all(np.diff(result.fitness_history) >= 0))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from src3.ga import GAConfig, GeneticAlgorithm
from src3.tsp.crossover import order_crossover
from src3.tsp.fitness import DistanceModel
from src3.tsp.mutation import swap_mutation
from src3.tsp.point import PointDataset, load_points
from src3.tsp.population import create_population
from src3.tsp.route import is_valid_route


def line_dataset(size: int = 31) -> PointDataset:
    points = np.zeros((size, 3), dtype=float)
    points[:, 0] = np.arange(1, size + 1)
    return PointDataset(origin=np.zeros(3), points=points, group=1)


class TspTests(unittest.TestCase):
    def test_loads_a_complete_group_from_assignment_csv(self) -> None:
        dataset = load_points("CaixeiroGruposGA.csv", group=1)
        self.assertEqual(dataset.size, 40)
        np.testing.assert_array_equal(dataset.origin, np.zeros(3))

    def test_route_distance_includes_return_to_origin(self) -> None:
        dataset = line_dataset()
        model = DistanceModel(dataset)
        route = np.arange(dataset.size)
        self.assertEqual(model.route_distance(route), 62.0)

    def test_population_contains_valid_permutations(self) -> None:
        population = create_population(10, 31, np.random.default_rng(1))
        self.assertTrue(all(is_valid_route(route, 31) for route in population))

    def test_order_crossover_keeps_valid_permutations(self) -> None:
        first = np.arange(40)
        second = first[::-1]
        for seed in range(20):
            children = order_crossover(
                first,
                second,
                np.random.default_rng(seed),
            )
            self.assertTrue(all(is_valid_route(child, 40) for child in children))

    def test_swap_mutation_changes_two_positions(self) -> None:
        route = np.arange(40)
        mutated = swap_mutation(route, np.random.default_rng(5))
        self.assertEqual(np.count_nonzero(route != mutated), 2)
        self.assertTrue(is_valid_route(mutated, 40))


class GeneticAlgorithmTests(unittest.TestCase):
    def test_ga_returns_a_valid_route(self) -> None:
        model = DistanceModel(line_dataset())
        config = GAConfig(
            population_size=20,
            max_generations=10,
            elite_size=2,
            stall_limit=5,
        )
        result = GeneticAlgorithm(
            model,
            config,
            np.random.default_rng(10),
        ).run()
        self.assertTrue(is_valid_route(result.best.chromosome, model.point_count))
        self.assertGreater(result.best.distance, 0.0)

    def test_elitism_prevents_generation_best_from_worsening(self) -> None:
        model = DistanceModel(line_dataset())
        config = GAConfig(
            population_size=20,
            max_generations=15,
            elite_size=2,
            stall_limit=15,
        )
        result = GeneticAlgorithm(
            model,
            config,
            np.random.default_rng(22),
        ).run()
        best_distances = np.asarray(
            [statistics.best_distance for statistics in result.history]
        )
        self.assertTrue(np.all(np.diff(best_distances) <= 0.0))


if __name__ == "__main__":
    unittest.main()

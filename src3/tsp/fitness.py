import numpy as np

from src3.tsp.point import PointDataset
from src3.tsp.route import validate_route


class DistanceModel:
    def __init__(self, dataset: PointDataset) -> None:
        self.dataset = dataset
        all_points = np.vstack((dataset.origin, dataset.points))
        differences = all_points[:, None, :] - all_points[None, :, :]
        self.distance_matrix = np.linalg.norm(differences, axis=2)

    @property
    def point_count(self) -> int:
        return self.dataset.size

    def route_distance(self, route: np.ndarray) -> float:
        route = validate_route(route, self.point_count)
        matrix_indices = route + 1
        distance = self.distance_matrix[0, matrix_indices[0]]
        distance += np.sum(
            self.distance_matrix[matrix_indices[:-1], matrix_indices[1:]]
        )
        distance += self.distance_matrix[matrix_indices[-1], 0]
        return float(distance)

    def population_distances(self, chromosomes: np.ndarray) -> np.ndarray:
        chromosomes = np.asarray(chromosomes, dtype=np.int32)
        if chromosomes.ndim != 2 or chromosomes.shape[1] != self.point_count:
            raise ValueError("Population has an invalid chromosome shape.")
        matrix_indices = chromosomes + 1
        distances = self.distance_matrix[0, matrix_indices[:, 0]].copy()
        distances += np.sum(
            self.distance_matrix[
                matrix_indices[:, :-1],
                matrix_indices[:, 1:],
            ],
            axis=1,
        )
        distances += self.distance_matrix[matrix_indices[:, -1], 0]
        return distances

    @staticmethod
    def fitness_from_distance(distance: float | np.ndarray) -> float | np.ndarray:
        return 1.0 / (1.0 + distance)

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Individual:
    chromosome: np.ndarray
    distance: float

    @property
    def fitness(self) -> float:
        return 1.0 / (1.0 + self.distance)


def create_population(
    population_size: int,
    point_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if point_count < 2:
        raise ValueError("point_count must be at least 2.")
    return np.asarray(
        [rng.permutation(point_count) for _ in range(population_size)],
        dtype=np.int32,
    )

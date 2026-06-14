import numpy as np


def tournament_selection(
    population: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
    *,
    tournament_size: int = 3,
) -> np.ndarray:
    if not 2 <= tournament_size <= len(population):
        raise ValueError("Invalid tournament_size.")
    competitors = rng.choice(
        len(population),
        size=tournament_size,
        replace=False,
    )
    winner = competitors[np.argmin(distances[competitors])]
    return population[winner].copy()

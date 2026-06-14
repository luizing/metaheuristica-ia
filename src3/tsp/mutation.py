import numpy as np


def swap_mutation(
    chromosome: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    chromosome = np.asarray(chromosome, dtype=np.int32).copy()
    first, second = rng.choice(len(chromosome), size=2, replace=False)
    chromosome[first], chromosome[second] = (
        chromosome[second],
        chromosome[first],
    )
    return chromosome

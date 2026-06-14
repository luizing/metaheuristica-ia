from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    x_best: np.ndarray
    best_value: float
    iterations: int
    evaluations: int
    history: np.ndarray


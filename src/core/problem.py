from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    objective_function: Callable[[np.ndarray], float]
    bounds: tuple[tuple[float, float], ...]
    maximize: bool
    known_optimum: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.bounds:
            raise ValueError("A problem must have at least one dimension.")
        if any(lower >= upper for lower, upper in self.bounds):
            raise ValueError("Every lower bound must be smaller than its upper bound.")
        if self.known_optimum is not None and len(self.known_optimum) != self.dimension:
            raise ValueError("known_optimum must match the problem dimension.")

    @property
    def dimension(self) -> int:
        return len(self.bounds)

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.array([bound[0] for bound in self.bounds], dtype=float)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array([bound[1] for bound in self.bounds], dtype=float)

    def clip(self, candidate: np.ndarray) -> np.ndarray:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != (self.dimension,):
            raise ValueError(f"Expected candidate with shape ({self.dimension},).")
        return np.clip(candidate, self.lower_bounds, self.upper_bounds)

    def evaluate(self, candidate: np.ndarray) -> float:
        candidate = self.clip(candidate)
        value = float(self.objective_function(candidate))
        if not np.isfinite(value):
            raise ValueError(f"{self.name} returned a non-finite objective value.")
        return value

    def is_better(self, candidate_value: float, current_value: float) -> bool:
        return (
            candidate_value > current_value
            if self.maximize
            else candidate_value < current_value
        )

    def best_index(self, values: np.ndarray) -> int:
        return int(np.argmax(values) if self.maximize else np.argmin(values))

    def worst_index(self, values: np.ndarray) -> int:
        return int(np.argmin(values) if self.maximize else np.argmax(values))

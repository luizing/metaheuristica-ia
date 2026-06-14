from collections.abc import Iterable, Iterator

import numpy as np

from src2.queens.board import BOARD_SIZE, validate_state
from src2.queens.objective import is_solution


class SolutionRepository:
    def __init__(self, solutions: Iterable[Iterable[int]] | None = None) -> None:
        self._solutions: set[tuple[int, ...]] = set()
        if solutions is not None:
            for solution in solutions:
                self.add(np.asarray(tuple(solution), dtype=np.int8))

    def add(self, state: np.ndarray) -> bool:
        state = validate_state(state)
        if not is_solution(state):
            raise ValueError("Only valid eight queens solutions can be stored.")
        solution = tuple(int(row) for row in state)
        previous_size = len(self._solutions)
        self._solutions.add(solution)
        return len(self._solutions) > previous_size

    def __contains__(self, state: object) -> bool:
        try:
            solution = tuple(int(row) for row in state)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return solution in self._solutions

    def __len__(self) -> int:
        return len(self._solutions)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(sorted(self._solutions))

    def as_array(self) -> np.ndarray:
        if not self._solutions:
            return np.empty((0, BOARD_SIZE), dtype=np.int8)
        return np.asarray(list(self), dtype=np.int8)

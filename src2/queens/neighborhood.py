import numpy as np

from src2.queens.board import BOARD_SIZE, validate_state


def perturb(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    current = validate_state(state)
    candidate = current.copy()
    column = int(rng.integers(0, BOARD_SIZE))
    old_row = int(candidate[column])

    new_row = int(rng.integers(1, BOARD_SIZE))
    if new_row >= old_row:
        new_row += 1
    candidate[column] = new_row
    return candidate

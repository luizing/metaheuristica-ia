import numpy as np

BOARD_SIZE = 8
MAX_PAIRS = BOARD_SIZE * (BOARD_SIZE - 1) // 2


def validate_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.int8)
    if state.shape != (BOARD_SIZE,):
        raise ValueError(f"The state must contain exactly {BOARD_SIZE} rows.")
    if np.any(state < 1) or np.any(state > BOARD_SIZE):
        raise ValueError(f"Every row must be between 1 and {BOARD_SIZE}.")
    return state


def random_state(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(1, BOARD_SIZE + 1, size=BOARD_SIZE, dtype=np.int8)


def format_board(state: np.ndarray) -> str:
    state = validate_state(state)
    rows = []
    for row in range(1, BOARD_SIZE + 1):
        rows.append(
            " ".join("Q" if queen_row == row else "." for queen_row in state)
        )
    return "\n".join(rows)

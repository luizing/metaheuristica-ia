import numpy as np

from src2.queens.board import BOARD_SIZE, MAX_PAIRS, validate_state


def attacking_pairs(state: np.ndarray) -> int:
    state = validate_state(state)
    attacks = 0
    for first_column in range(BOARD_SIZE - 1):
        for second_column in range(first_column + 1, BOARD_SIZE):
            same_row = state[first_column] == state[second_column]
            same_diagonal = abs(
                int(state[first_column]) - int(state[second_column])
            ) == second_column - first_column
            if same_row or same_diagonal:
                attacks += 1
    return attacks


def fitness(state: np.ndarray) -> int:
    return MAX_PAIRS - attacking_pairs(state)


def is_solution(state: np.ndarray) -> bool:
    return fitness(state) == MAX_PAIRS

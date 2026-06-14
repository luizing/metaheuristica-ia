from dataclasses import dataclass
from math import exp
from time import perf_counter

import numpy as np

from src2.annealing.cooling import GeometricCooling
from src2.queens.board import MAX_PAIRS, random_state, validate_state
from src2.queens.neighborhood import perturb
from src2.queens.objective import fitness


@dataclass(frozen=True)
class AnnealingResult:
    state: np.ndarray
    fitness: int
    success: bool
    iterations: int
    final_temperature: float
    elapsed_seconds: float
    fitness_history: np.ndarray
    temperature_history: np.ndarray


def simulated_annealing(
    rng: np.random.Generator,
    *,
    initial_temperature: float = 100.0,
    alpha: float = 0.99,
    max_iterations: int = 10_000,
    initial_state: np.ndarray | None = None,
    record_history: bool = True,
) -> AnnealingResult:
    if initial_temperature <= 0.0:
        raise ValueError("initial_temperature must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    cooling = GeometricCooling(alpha=alpha)
    current = (
        random_state(rng)
        if initial_state is None
        else validate_state(initial_state).copy()
    )
    current_fitness = fitness(current)
    best = current.copy()
    best_fitness = current_fitness
    temperature = float(initial_temperature)
    fitness_history = [current_fitness] if record_history else []
    temperature_history = [temperature] if record_history else []
    start = perf_counter()

    if current_fitness == MAX_PAIRS:
        iterations = 0
    else:
        for iterations in range(1, max_iterations + 1):
            candidate = perturb(current, rng)
            candidate_fitness = fitness(candidate)
            delta = candidate_fitness - current_fitness

            if delta >= 0 or rng.random() < exp(delta / temperature):
                current = candidate
                current_fitness = candidate_fitness

            if current_fitness > best_fitness:
                best = current.copy()
                best_fitness = current_fitness

            temperature = cooling.cool(temperature)
            if record_history:
                fitness_history.append(best_fitness)
                temperature_history.append(temperature)

            if current_fitness == MAX_PAIRS:
                best = current.copy()
                best_fitness = current_fitness
                break

    elapsed = perf_counter() - start
    return AnnealingResult(
        state=best,
        fitness=best_fitness,
        success=best_fitness == MAX_PAIRS,
        iterations=iterations,
        final_temperature=temperature,
        elapsed_seconds=elapsed,
        fitness_history=np.asarray(fitness_history, dtype=np.int8),
        temperature_history=np.asarray(temperature_history, dtype=float),
    )

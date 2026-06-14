from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src3.tsp.crossover import order_crossover
from src3.tsp.fitness import DistanceModel
from src3.tsp.mutation import swap_mutation
from src3.tsp.population import Individual, create_population
from src3.tsp.route import validate_route
from src3.tsp.selection import tournament_selection


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    max_generations: int = 500
    crossover_rate: float = 0.9
    mutation_rate: float = 0.01
    tournament_size: int = 3
    elite_size: int = 5
    stall_limit: int = 50

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.max_generations <= 0:
            raise ValueError("max_generations must be positive.")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("Invalid tournament_size.")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be smaller than population_size.")
        if self.stall_limit <= 0:
            raise ValueError("stall_limit must be positive.")


@dataclass(frozen=True)
class GenerationStatistics:
    generation: int
    best_distance: float
    average_distance: float
    worst_distance: float
    best_fitness: float
    average_fitness: float
    worst_fitness: float


@dataclass(frozen=True)
class GAResult:
    best: Individual
    generations: int
    generations_to_acceptable: int | None
    acceptable_distance: float | None
    elapsed_seconds: float
    history: tuple[GenerationStatistics, ...]


class GeneticAlgorithm:
    def __init__(
        self,
        distance_model: DistanceModel,
        config: GAConfig,
        rng: np.random.Generator,
    ) -> None:
        self.distance_model = distance_model
        self.config = config
        self.rng = rng

    def run(self, *, acceptable_distance: float | None = None) -> GAResult:
        if acceptable_distance is not None and acceptable_distance <= 0:
            raise ValueError("acceptable_distance must be positive.")

        population = create_population(
            self.config.population_size,
            self.distance_model.point_count,
            self.rng,
        )
        distances = self.distance_model.population_distances(population)
        best_index = int(np.argmin(distances))
        global_best = Individual(
            chromosome=population[best_index].copy(),
            distance=float(distances[best_index]),
        )
        generations_to_acceptable = (
            0
            if acceptable_distance is not None
            and global_best.distance <= acceptable_distance
            else None
        )
        history = [self._statistics(0, distances)]
        stall = 0
        start = perf_counter()

        for generation in range(1, self.config.max_generations + 1):
            population = self._next_generation(population, distances)
            distances = self.distance_model.population_distances(population)
            best_index = int(np.argmin(distances))
            generation_best_distance = float(distances[best_index])

            if generation_best_distance < global_best.distance:
                global_best = Individual(
                    chromosome=population[best_index].copy(),
                    distance=generation_best_distance,
                )
                stall = 0
            else:
                stall += 1

            if (
                generations_to_acceptable is None
                and acceptable_distance is not None
                and global_best.distance <= acceptable_distance
            ):
                generations_to_acceptable = generation

            history.append(self._statistics(generation, distances))
            if stall >= self.config.stall_limit:
                break

        validate_route(global_best.chromosome, self.distance_model.point_count)
        return GAResult(
            best=global_best,
            generations=generation,
            generations_to_acceptable=generations_to_acceptable,
            acceptable_distance=acceptable_distance,
            elapsed_seconds=perf_counter() - start,
            history=tuple(history),
        )

    def _next_generation(
        self,
        population: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        elite_indices = np.argsort(distances)[: self.config.elite_size]
        offspring = [population[index].copy() for index in elite_indices]

        while len(offspring) < self.config.population_size:
            first_parent = tournament_selection(
                population,
                distances,
                self.rng,
                tournament_size=self.config.tournament_size,
            )
            second_parent = tournament_selection(
                population,
                distances,
                self.rng,
                tournament_size=self.config.tournament_size,
            )

            if self.rng.random() < self.config.crossover_rate:
                first_child, second_child = order_crossover(
                    first_parent,
                    second_parent,
                    self.rng,
                )
            else:
                first_child, second_child = first_parent, second_parent

            if self.rng.random() < self.config.mutation_rate:
                first_child = swap_mutation(first_child, self.rng)
            if self.rng.random() < self.config.mutation_rate:
                second_child = swap_mutation(second_child, self.rng)

            offspring.append(first_child)
            if len(offspring) < self.config.population_size:
                offspring.append(second_child)

        return np.asarray(offspring, dtype=np.int32)

    @staticmethod
    def _statistics(
        generation: int,
        distances: np.ndarray,
    ) -> GenerationStatistics:
        fitness = DistanceModel.fitness_from_distance(distances)
        return GenerationStatistics(
            generation=generation,
            best_distance=float(np.min(distances)),
            average_distance=float(np.mean(distances)),
            worst_distance=float(np.max(distances)),
            best_fitness=float(np.max(fitness)),
            average_fitness=float(np.mean(fitness)),
            worst_fitness=float(np.min(fitness)),
        )

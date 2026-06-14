import numpy as np

from src.core.problem import Problem


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def gaussian_peaks(x: np.ndarray) -> float:
    x1, x2 = x
    return float(
        np.exp(-(x1**2 + x2**2))
        + 2.0 * np.exp(-((x1 - 1.7) ** 2 + (x2 - 1.7) ** 2))
    )


def ackley(x: np.ndarray) -> float:
    squared_mean = np.mean(x**2)
    cosine_mean = np.mean(np.cos(2.0 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(squared_mean))
        - np.exp(cosine_mean)
        + 20.0
        + np.e
    )


def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))


def damped_gaussian(x: np.ndarray) -> float:
    x1, x2 = x
    return float(
        (x1 * np.cos(x1)) / 20.0
        + 2.0 * np.exp(-(x1**2) - (x2 - 1.0) ** 2)
        + 0.01 * x1 * x2
    )


def sinusoidal(x: np.ndarray) -> float:
    x1, x2 = x
    return float(
        x1 * np.sin(4.0 * np.pi * x1)
        - x2 * np.sin(4.0 * np.pi * x2 + np.pi)
        + 1.0
    )


PROBLEMS = (
    Problem(
        name="F1 - Sphere",
        objective_function=sphere,
        bounds=((-100.0, 100.0), (-100.0, 100.0)),
        maximize=False,
        known_optimum=(0.0, 0.0),
    ),
    Problem(
        name="F2 - Gaussian Peaks",
        objective_function=gaussian_peaks,
        bounds=((-2.0, 4.0), (-2.0, 5.0)),
        maximize=True,
    ),
    Problem(
        name="F3 - Ackley",
        objective_function=ackley,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        maximize=False,
        known_optimum=(0.0, 0.0),
    ),
    Problem(
        name="F4 - Rastrigin",
        objective_function=rastrigin,
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        maximize=False,
        known_optimum=(0.0, 0.0),
    ),
    Problem(
        name="F5 - Damped Gaussian",
        objective_function=damped_gaussian,
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        maximize=True,
    ),
    Problem(
        name="F6 - Sinusoidal",
        objective_function=sinusoidal,
        bounds=((-1.0, 3.0), (-1.0, 3.0)),
        maximize=True,
    ),
)


def get_problem(identifier: str) -> Problem:
    normalized = identifier.strip().lower()
    for index, problem in enumerate(PROBLEMS, start=1):
        if normalized in {str(index), f"f{index}", problem.name.lower()}:
            return problem
    raise KeyError(f"Unknown problem: {identifier}")

import numpy as np


def is_valid_route(route: np.ndarray, point_count: int) -> bool:
    route = np.asarray(route)
    if route.shape != (point_count,):
        return False
    return np.array_equal(np.sort(route), np.arange(point_count))


def validate_route(route: np.ndarray, point_count: int) -> np.ndarray:
    route = np.asarray(route, dtype=np.int32)
    if not is_valid_route(route, point_count):
        raise ValueError("A route must be a permutation of all point indices.")
    return route

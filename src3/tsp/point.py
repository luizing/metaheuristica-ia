from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Point3D:
    identifier: int
    x: float
    y: float
    z: float

    @property
    def coordinates(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True)
class PointDataset:
    origin: np.ndarray
    points: np.ndarray
    group: int

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float)
        points = np.asarray(self.points, dtype=float)
        if origin.shape != (3,):
            raise ValueError("The origin must have three coordinates.")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape (n, 3).")
        if not np.isfinite(origin).all() or not np.isfinite(points).all():
            raise ValueError("Coordinates must be finite.")
        if not 30 < len(points) < 60:
            raise ValueError("The assignment requires between 31 and 59 points.")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "points", points)

    @property
    def size(self) -> int:
        return len(self.points)


def load_points(csv_path: str | Path, *, group: int = 1) -> PointDataset:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] not in (3, 4):
        raise ValueError("The CSV must contain x,y,z or x,y,z,group columns.")

    if data.shape[1] == 3:
        origin = np.zeros(3, dtype=float)
        points = data
    else:
        group_column = data[:, 3].astype(int)
        origin_rows = data[group_column == 0, :3]
        origin = origin_rows[0] if len(origin_rows) else np.zeros(3, dtype=float)
        points = data[group_column == group, :3]
        if len(points) == 0:
            available = sorted(set(group_column) - {0})
            raise ValueError(f"Group {group} was not found. Available groups: {available}")

    return PointDataset(origin=origin, points=points, group=group)

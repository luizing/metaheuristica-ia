from dataclasses import dataclass


@dataclass(frozen=True)
class GeometricCooling:
    alpha: float = 0.99
    minimum_temperature: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if self.minimum_temperature <= 0.0:
            raise ValueError("minimum_temperature must be positive.")

    def cool(self, temperature: float) -> float:
        return max(self.minimum_temperature, self.alpha * temperature)

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass
class OnlineStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return sqrt(self.m2 / (self.count - 1))

    def threshold(self, sigma: float = 3.0, floor: float = 0.0) -> float:
        return max(floor, self.mean + sigma * self.std)


class CalibrationStore:
    """Collects clean-run residuals and produces mu + k*sigma thresholds."""

    def __init__(self) -> None:
        self._stats: dict[str, OnlineStats] = {}

    def add(self, name: str, residual: float) -> None:
        self._stats.setdefault(name, OnlineStats()).add(residual)

    def thresholds(self, sigma: float = 3.0, floors: dict[str, float] | None = None) -> dict[str, float]:
        floors = floors or {}
        return {
            name: stats.threshold(sigma=sigma, floor=floors.get(name, 0.0))
            for name, stats in self._stats.items()
        }

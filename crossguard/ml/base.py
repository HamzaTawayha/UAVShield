from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from crossguard.defense.state import DroneState


@dataclass(frozen=True)
class MLSanityDecision:
    """Decision returned by a learned state-consistency checker."""

    ready: bool
    alert: bool = False
    score: float | None = None
    threshold: float | None = None
    backend: str = "unknown"
    reason: str = ""


class StateSanityChecker(Protocol):
    """Small interface used by CrossGuardHarness for learned checks."""

    def observe(self, state: DroneState) -> MLSanityDecision:
        ...

    def reset(self) -> None:
        ...

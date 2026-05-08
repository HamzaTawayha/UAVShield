from __future__ import annotations

from dataclasses import dataclass

from crossguard.defense.state import Violation


@dataclass
class ScoreState:
    suspicion: int = 0
    alert: bool = False


class ViolationScorer:
    """Accumulates transient invariant failures into stable alerts."""

    def __init__(self, alert_threshold: int = 3, decay: int = 1) -> None:
        self.alert_threshold = alert_threshold
        self.decay = decay
        self.state = ScoreState()

    def update(self, violations: list[Violation]) -> ScoreState:
        if violations:
            self.state.suspicion += sum(max(1, v.severity) for v in violations)
        else:
            self.state.suspicion = max(0, self.state.suspicion - self.decay)

        self.state.alert = self.state.suspicion >= self.alert_threshold
        if self.state.alert:
            self.state.suspicion = 0
        return ScoreState(suspicion=self.state.suspicion, alert=self.state.alert)

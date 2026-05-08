from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from crossguard.defense.invariants import InvariantConfig, run_all_checks
from crossguard.defense.scoring import ScoreState, ViolationScorer
from crossguard.defense.state import DroneState, Violation


@dataclass(frozen=True)
class HarnessDecision:
    state: DroneState
    violations: tuple[Violation, ...]
    suspicion: int
    alert: bool

    @property
    def is_valid(self) -> bool:
        return not self.alert and not self.violations


class CrossGuardHarness:
    """Pure-Python sanity harness independent of ROS2 or MAVSDK."""

    def __init__(
        self,
        config: InvariantConfig | None = None,
        history_size: int = 300,
        alert_threshold: int = 3,
    ) -> None:
        self.config = config or InvariantConfig()
        self.history: deque[DroneState] = deque(maxlen=history_size)
        self.scorer = ViolationScorer(alert_threshold=alert_threshold)

    def observe(self, state: DroneState) -> HarnessDecision:
        self.history.append(state)
        history = list(self.history)
        violations = tuple(run_all_checks(history, self.config))
        score: ScoreState = self.scorer.update(list(violations))
        return HarnessDecision(
            state=state,
            violations=violations,
            suspicion=score.suspicion,
            alert=score.alert,
        )

    def reset(self) -> None:
        self.history.clear()
        self.scorer = ViolationScorer(
            alert_threshold=self.scorer.alert_threshold,
            decay=self.scorer.decay,
        )

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from crossguard.defense.invariants import InvariantConfig, run_all_checks
from crossguard.defense.scoring import ScoreState, ViolationScorer
from crossguard.defense.state import DroneState, Violation
from crossguard.ml.base import MLSanityDecision, StateSanityChecker


@dataclass(frozen=True)
class HarnessDecision:
    state: DroneState
    violations: tuple[Violation, ...]
    suspicion: int
    alert: bool
    ml_decision: MLSanityDecision | None = None

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
        ml_checker: StateSanityChecker | None = None,
    ) -> None:
        self.config = config or InvariantConfig()
        self.history: deque[DroneState] = deque(maxlen=history_size)
        self.scorer = ViolationScorer(alert_threshold=alert_threshold)
        self.ml_checker = ml_checker

    def observe(self, state: DroneState) -> HarnessDecision:
        self.history.append(state)
        history = list(self.history)
        violations = list(run_all_checks(history, self.config))
        ml_decision = self.ml_checker.observe(state) if self.ml_checker is not None else None
        if ml_decision is not None and ml_decision.ready and ml_decision.alert:
            violations.append(
                Violation(
                    "ml.state_anomaly",
                    3,
                    "learned UAV state model flagged this rolling state window",
                    observed=ml_decision.score,
                    threshold=ml_decision.threshold,
                    timestamp=state.timestamp,
                )
            )
        score: ScoreState = self.scorer.update(list(violations))
        alert = score.alert or bool(ml_decision and ml_decision.ready and ml_decision.alert)
        return HarnessDecision(
            state=state,
            violations=tuple(violations),
            suspicion=score.suspicion,
            alert=alert,
            ml_decision=ml_decision,
        )

    def reset(self) -> None:
        self.history.clear()
        self.scorer = ViolationScorer(
            alert_threshold=self.scorer.alert_threshold,
            decay=self.scorer.decay,
        )
        if self.ml_checker is not None:
            self.ml_checker.reset()

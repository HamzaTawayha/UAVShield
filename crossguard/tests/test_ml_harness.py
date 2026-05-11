from __future__ import annotations

import unittest

from crossguard.defense.harness import CrossGuardHarness
from crossguard.defense.state import DroneState, GeoPoint
from crossguard.ml.base import MLSanityDecision


class FakeMlChecker:
    def __init__(self, decision: MLSanityDecision) -> None:
        self.decision = decision
        self.reset_called = False

    def observe(self, state: DroneState) -> MLSanityDecision:
        return self.decision

    def reset(self) -> None:
        self.reset_called = True


class CrossGuardMlHarnessTests(unittest.TestCase):
    def test_ml_alert_adds_state_anomaly_violation(self) -> None:
        harness = CrossGuardHarness(
            ml_checker=FakeMlChecker(
                MLSanityDecision(
                    ready=True,
                    alert=True,
                    score=0.91,
                    threshold=0.80,
                    backend="test",
                )
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=1.0,
                position=GeoPoint(0.0, 0.0, 5.0),
            )
        )

        self.assertTrue(decision.alert)
        self.assertIsNotNone(decision.ml_decision)
        self.assertIn("ml.state_anomaly", {violation.check_id for violation in decision.violations})

    def test_ml_warmup_does_not_create_violation(self) -> None:
        harness = CrossGuardHarness(
            ml_checker=FakeMlChecker(
                MLSanityDecision(
                    ready=False,
                    alert=False,
                    backend="test",
                    reason="warming up",
                )
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=1.0,
                position=GeoPoint(0.0, 0.0, 5.0),
            )
        )

        self.assertFalse(decision.alert)
        self.assertEqual([], list(decision.violations))

    def test_reset_resets_ml_checker(self) -> None:
        checker = FakeMlChecker(MLSanityDecision(ready=False))
        harness = CrossGuardHarness(ml_checker=checker)

        harness.reset()

        self.assertTrue(checker.reset_called)


if __name__ == "__main__":
    unittest.main()

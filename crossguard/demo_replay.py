from __future__ import annotations

from crossguard.defense.harness import CrossGuardHarness
from crossguard.defense.state import BatteryState, DroneState, GeoPoint, PerceptionObject, Velocity


def main() -> None:
    harness = CrossGuardHarness()
    samples = [
        DroneState(
            timestamp=0.0,
            position=GeoPoint(42.3314, -83.0458, 20.0),
            velocity=Velocity(0.0, 0.0, 0.0),
            battery=BatteryState(100.0),
        ),
        DroneState(
            timestamp=240.0,
            position=GeoPoint(42.3315, -83.0459, 20.0),
            velocity=Velocity(0.0, 0.0, 0.0),
            battery=BatteryState(2.0),
        ),
        DroneState(
            timestamp=1_440.0,
            position=GeoPoint(34.0522, -118.2437, 20.0),
            velocity=Velocity(0.0, 0.0, 0.0),
            battery=BatteryState(2.0),
            perception_objects=(
                PerceptionObject(
                    class_id="obstacle",
                    bbox_center_x=320,
                    bbox_center_y=240,
                    confidence=0.92,
                    claimed_range_m=3.0,
                    depth_range_m=15.0,
                ),
            ),
        ),
    ]

    for sample in samples:
        decision = harness.observe(sample)
        print(f"t={sample.timestamp:.0f}s alert={decision.alert} suspicion={decision.suspicion}")
        for violation in decision.violations:
            print(f"  - {violation.check_id}: {violation.message}")


if __name__ == "__main__":
    main()

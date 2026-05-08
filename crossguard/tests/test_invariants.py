from __future__ import annotations

import unittest

from crossguard.defense.harness import CrossGuardHarness
from crossguard.defense.invariants import InvariantConfig
from crossguard.defense.state import (
    Acceleration,
    Attitude,
    BatteryState,
    Command,
    DroneState,
    GeoPoint,
    PeerDroneState,
    PerceptionObject,
    Velocity,
)


def violation_ids(decision) -> set[str]:
    return {violation.check_id for violation in decision.violations}


class CrossGuardInvariantTests(unittest.TestCase):
    def test_impossible_battery_drop_triggers_alert(self) -> None:
        harness = CrossGuardHarness()
        harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(42.3314, -83.0458, 10.0),
                battery=BatteryState(100.0),
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=240.0,
                position=GeoPoint(42.3314, -83.0458, 10.0),
                battery=BatteryState(2.0),
            )
        )

        self.assertTrue(decision.alert)
        self.assertIn("battery.impossible_drop", violation_ids(decision))

    def test_impossible_location_jump_triggers_alert(self) -> None:
        harness = CrossGuardHarness()
        harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(42.3314, -83.0458, 20.0),
                velocity=Velocity(0.0, 0.0, 0.0),
                battery=BatteryState(100.0),
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=1_200.0,
                position=GeoPoint(34.0522, -118.2437, 20.0),
                velocity=Velocity(0.0, 0.0, 0.0),
                battery=BatteryState(99.0),
            )
        )

        self.assertTrue(decision.alert)
        self.assertIn("gps.impossible_jump", violation_ids(decision))

    def test_perception_depth_mismatch_triggers_alert(self) -> None:
        harness = CrossGuardHarness()
        decision = harness.observe(
            DroneState(
                timestamp=10.0,
                position=GeoPoint(42.3314, -83.0458, 20.0),
                perception_objects=(
                    PerceptionObject(
                        class_id="obstacle",
                        bbox_center_x=320.0,
                        bbox_center_y=240.0,
                        confidence=0.92,
                        claimed_range_m=3.0,
                        depth_range_m=15.2,
                    ),
                ),
            )
        )

        self.assertTrue(decision.alert)
        self.assertIn("perception.depth_mismatch", violation_ids(decision))

    def test_false_waypoint_reached_triggers_alert(self) -> None:
        harness = CrossGuardHarness()
        decision = harness.observe(
            DroneState(
                timestamp=10.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                current_waypoint=GeoPoint(0.0, 0.0001, 5.0),
                reported_waypoint_reached=True,
            )
        )

        self.assertTrue(decision.alert)
        self.assertIn("mission.false_waypoint_reached", violation_ids(decision))

    def test_command_without_progress_is_flagged(self) -> None:
        harness = CrossGuardHarness()
        command = Command(
            command_type="goto",
            issued_at=0.0,
            target=GeoPoint(0.0, 10.0 / 111_139.0, 5.0),
        )
        harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                last_command=command,
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=6.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                last_command=command,
            )
        )

        self.assertFalse(decision.alert)
        self.assertIn("command.no_effect", violation_ids(decision))

    def test_normal_small_motion_passes(self) -> None:
        harness = CrossGuardHarness(config=InvariantConfig(max_battery_drop_pct_per_min=10.0))
        harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(42.3314, -83.0458, 20.0),
                velocity=Velocity(1.0, 0.0, 0.0),
                battery=BatteryState(100.0),
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=10.0,
                position=GeoPoint(42.3314, -83.04568, 20.0),
                velocity=Velocity(1.0, 0.0, 0.0),
                battery=BatteryState(99.5),
            )
        )

        self.assertFalse(decision.alert)
        self.assertEqual([], list(decision.violations))

    def test_imu_velocity_mismatch_is_flagged(self) -> None:
        harness = CrossGuardHarness()
        harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                velocity=Velocity(0.0, 0.0, 0.0),
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=1.0,
                position=GeoPoint(0.0, 0.00001, 5.0),
                velocity=Velocity(10.0, 0.0, 0.0),
                acceleration=Acceleration(0.0, 0.0, 0.0),
            )
        )

        self.assertIn("imu.velocity_mismatch", violation_ids(decision))

    def test_heading_velocity_mismatch_is_flagged(self) -> None:
        harness = CrossGuardHarness()
        harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                velocity=Velocity(0.0, 5.0, 0.0),
                attitude=Attitude(0.0, 0.0, 90.0),
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=1.0,
                position=GeoPoint(0.0, 0.000045, 5.0),
                velocity=Velocity(0.0, 5.0, 0.0),
                attitude=Attitude(0.0, 0.0, 180.0),
            )
        )

        self.assertIn("compass.heading_velocity_mismatch", violation_ids(decision))

    def test_stale_sensor_timestamp_is_flagged(self) -> None:
        harness = CrossGuardHarness()
        decision = harness.observe(
            DroneState(
                timestamp=10.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                sensor_timestamps={"gps": 6.0, "battery": 9.8},
            )
        )

        self.assertIn("sensor.stale", violation_ids(decision))

    def test_replayed_packet_id_triggers_alert(self) -> None:
        harness = CrossGuardHarness()
        harness.observe(
            DroneState(
                timestamp=0.0,
                packet_id="abc",
                position=GeoPoint(0.0, 0.0, 5.0),
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=1.0,
                packet_id="abc",
                position=GeoPoint(0.0, 0.0, 5.0),
            )
        )

        self.assertTrue(decision.alert)
        self.assertIn("packet.replay", violation_ids(decision))

    def test_geofence_violation_triggers_alert(self) -> None:
        harness = CrossGuardHarness(
            config=InvariantConfig(
                geofence_min_lat=42.0,
                geofence_max_lat=43.0,
                geofence_min_lon=-84.0,
                geofence_max_lon=-83.0,
                geofence_min_alt_m=0.0,
                geofence_max_alt_m=120.0,
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(44.0, -83.5, 20.0),
            )
        )

        self.assertTrue(decision.alert)
        self.assertIn("geofence.state_lat", violation_ids(decision))

    def test_command_requiring_impossible_speed_triggers_alert(self) -> None:
        harness = CrossGuardHarness()
        command = Command(
            command_type="goto",
            issued_at=0.0,
            target=GeoPoint(0.0, 100.0 / 111_139.0, 5.0),
            metadata={"eta_s": 1.0},
        )

        decision = harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                last_command=command,
            )
        )

        self.assertTrue(decision.alert)
        self.assertIn("command.required_ground_speed", violation_ids(decision))

    def test_ground_range_mismatch_is_flagged(self) -> None:
        harness = CrossGuardHarness()
        decision = harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(0.0, 0.0, 20.0),
                ground_distance_m=5.0,
            )
        )

        self.assertIn("altitude.ground_range_mismatch", violation_ids(decision))

    def test_battery_current_drop_mismatch_is_flagged(self) -> None:
        harness = CrossGuardHarness()
        harness.observe(
            DroneState(
                timestamp=0.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                battery=BatteryState(percent=100.0, current_a=0.2),
            )
        )

        decision = harness.observe(
            DroneState(
                timestamp=60.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                battery=BatteryState(percent=98.0, current_a=0.2),
            )
        )

        self.assertIn("battery.current_drop_mismatch", violation_ids(decision))

    def test_peer_false_waypoint_reached_triggers_alert(self) -> None:
        harness = CrossGuardHarness()
        decision = harness.observe(
            DroneState(
                timestamp=10.0,
                position=GeoPoint(0.0, 0.0, 5.0),
                peer_states=(
                    PeerDroneState(
                        drone_id="drone2",
                        timestamp=9.8,
                        position=GeoPoint(0.0, 0.0, 5.0),
                        current_waypoint=GeoPoint(0.0, 0.0001, 5.0),
                        reported_waypoint_reached=True,
                    ),
                ),
            )
        )

        self.assertTrue(decision.alert)
        self.assertIn("peer.false_waypoint_reached", violation_ids(decision))


if __name__ == "__main__":
    unittest.main()

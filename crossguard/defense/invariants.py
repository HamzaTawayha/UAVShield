from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees
from typing import Iterable

from crossguard.defense.state import DroneState, Violation
from crossguard.utils.geo import (
    distance_3d_m,
    horizontal_distance_m,
    required_speed_mps,
    wrap_angle_delta_deg,
)


@dataclass(frozen=True)
class InvariantConfig:
    """Conservative defaults for a small UAV in simulation or lab flight."""

    max_ground_speed_mps: float = 35.0
    max_vertical_speed_mps: float = 8.0
    max_acceleration_mps2: float = 20.0
    max_imu_velocity_residual_mps2: float = 8.0
    velocity_position_tolerance_mps: float = 8.0
    max_battery_drop_pct_per_min: float = 5.0
    max_battery_rise_pct_per_min: float = 20.0
    low_current_threshold_a: float = 1.0
    max_low_current_battery_drop_pct_per_min: float = 1.0
    max_roll_pitch_rate_deg_s: float = 180.0
    max_yaw_rate_deg_s: float = 240.0
    min_heading_check_speed_mps: float = 2.0
    heading_velocity_tolerance_deg: float = 75.0
    perception_depth_tolerance_m: float = 2.0
    waypoint_radius_m: float = 3.0
    command_progress_window_s: float = 5.0
    command_min_progress_m: float = 0.5
    max_command_distance_m: float = 1_000.0
    max_sensor_age_s: float = 2.0
    max_clock_skew_s: float = 0.25
    terrain_alt_m: float = 0.0
    max_ground_altitude_residual_m: float = 3.0
    geofence_min_lat: float | None = None
    geofence_max_lat: float | None = None
    geofence_min_lon: float | None = None
    geofence_max_lon: float | None = None
    geofence_min_alt_m: float | None = None
    geofence_max_alt_m: float | None = None
    max_peer_state_age_s: float = 2.0
    min_peer_separation_m: float = 1.0
    min_dt_s: float = 1e-3
    calibrated_thresholds: dict[str, float] = field(default_factory=dict)

    def threshold(self, name: str, default: float) -> float:
        return self.calibrated_thresholds.get(name, default)


def run_all_checks(history: Iterable[DroneState], cfg: InvariantConfig) -> list[Violation]:
    states = list(history)
    if not states:
        return []

    violations: list[Violation] = []
    curr = states[-1]
    prev = states[-2] if len(states) >= 2 else None

    violations.extend(check_replay_or_duplicate(states, cfg))
    violations.extend(check_sensor_freshness(curr, cfg))
    violations.extend(check_geofence(curr, cfg))
    violations.extend(check_command_plausibility(curr, cfg))
    violations.extend(check_ground_altitude(curr, cfg))
    violations.extend(check_multi_drone_consistency(curr, cfg))

    if prev is not None:
        violations.extend(check_position_speed(prev, curr, cfg))
        violations.extend(check_altitude_rate(prev, curr, cfg))
        violations.extend(check_velocity_consistency(prev, curr, cfg))
        violations.extend(check_acceleration(prev, curr, cfg))
        violations.extend(check_imu_velocity_consistency(prev, curr, cfg))
        violations.extend(check_battery_rate(prev, curr, cfg))
        violations.extend(check_attitude_rate(prev, curr, cfg))
        violations.extend(check_heading_velocity(curr, cfg))

    violations.extend(check_perception_depth(curr, cfg))
    violations.extend(check_waypoint_claim(curr, cfg))
    violations.extend(check_command_effect(states, cfg))
    return violations


def _dt(prev: DroneState, curr: DroneState) -> float:
    return curr.timestamp - prev.timestamp


def _bad_timestamp(prev: DroneState, curr: DroneState) -> Violation | None:
    dt_s = _dt(prev, curr)
    if dt_s <= 0:
        return Violation(
            "time.monotonic",
            3,
            "non-monotonic state timestamps",
            observed=dt_s,
            threshold="> 0",
            timestamp=curr.timestamp,
        )
    return None


def check_position_speed(prev: DroneState, curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    bad_time = _bad_timestamp(prev, curr)
    if bad_time:
        return [bad_time]

    dt_s = _dt(prev, curr)
    speed = required_speed_mps(prev.position, curr.position, dt_s)
    threshold = cfg.threshold("max_ground_speed_mps", cfg.max_ground_speed_mps)
    if speed > threshold:
        return [
            Violation(
                "gps.impossible_jump",
                3,
                f"reported position requires {speed:.1f} m/s ground speed",
                observed=speed,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_altitude_rate(prev: DroneState, curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    bad_time = _bad_timestamp(prev, curr)
    if bad_time:
        return [bad_time]

    dt_s = _dt(prev, curr)
    vertical_rate = abs(curr.position.alt_m - prev.position.alt_m) / max(dt_s, cfg.min_dt_s)
    threshold = cfg.threshold("max_vertical_speed_mps", cfg.max_vertical_speed_mps)
    if vertical_rate > threshold:
        return [
            Violation(
                "altitude.rate",
                2,
                f"altitude changed at {vertical_rate:.1f} m/s",
                observed=vertical_rate,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_velocity_consistency(prev: DroneState, curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    if curr.velocity is None:
        return []

    bad_time = _bad_timestamp(prev, curr)
    if bad_time:
        return [bad_time]

    dt_s = _dt(prev, curr)
    position_speed = horizontal_distance_m(prev.position, curr.position) / max(dt_s, cfg.min_dt_s)
    reported_speed = curr.velocity.horizontal_mps
    residual = abs(position_speed - reported_speed)
    threshold = cfg.threshold("velocity_position_tolerance_mps", cfg.velocity_position_tolerance_mps)
    if residual > threshold:
        return [
            Violation(
                "telemetry.velocity_position_mismatch",
                2,
                f"position delta implies {position_speed:.1f} m/s but velocity reports {reported_speed:.1f} m/s",
                observed=residual,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_acceleration(prev: DroneState, curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    if prev.velocity is None or curr.velocity is None:
        return []

    bad_time = _bad_timestamp(prev, curr)
    if bad_time:
        return [bad_time]

    dt_s = _dt(prev, curr)
    dv = abs(curr.velocity.magnitude_mps - prev.velocity.magnitude_mps)
    accel = dv / max(dt_s, cfg.min_dt_s)
    threshold = cfg.threshold("max_acceleration_mps2", cfg.max_acceleration_mps2)
    if accel > threshold:
        return [
            Violation(
                "velocity.acceleration_spike",
                2,
                f"reported velocity changed at {accel:.1f} m/s^2",
                observed=accel,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_imu_velocity_consistency(prev: DroneState, curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    if prev.velocity is None or curr.velocity is None or curr.acceleration is None:
        return []

    bad_time = _bad_timestamp(prev, curr)
    if bad_time:
        return [bad_time]

    dt_s = max(_dt(prev, curr), cfg.min_dt_s)
    dvx = (curr.velocity.vx_mps - prev.velocity.vx_mps) / dt_s
    dvy = (curr.velocity.vy_mps - prev.velocity.vy_mps) / dt_s
    dvz = (curr.velocity.vz_mps - prev.velocity.vz_mps) / dt_s
    residual = (
        (dvx - curr.acceleration.ax_mps2) ** 2
        + (dvy - curr.acceleration.ay_mps2) ** 2
        + (dvz - curr.acceleration.az_mps2) ** 2
    ) ** 0.5
    threshold = cfg.threshold("max_imu_velocity_residual_mps2", cfg.max_imu_velocity_residual_mps2)
    if residual > threshold:
        return [
            Violation(
                "imu.velocity_mismatch",
                2,
                f"IMU acceleration disagrees with velocity delta by {residual:.1f} m/s^2",
                observed=residual,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_battery_rate(prev: DroneState, curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    if prev.battery is None or curr.battery is None:
        return []

    bad_time = _bad_timestamp(prev, curr)
    if bad_time:
        return [bad_time]

    dt_min = _dt(prev, curr) / 60.0
    if dt_min <= 0:
        return []

    prev_pct = prev.battery.percent
    curr_pct = curr.battery.percent
    if not (0.0 <= curr_pct <= 100.0):
        return [
            Violation(
                "battery.range",
                3,
                f"battery percentage {curr_pct:.1f} is outside 0-100",
                observed=curr_pct,
                threshold="0..100",
                timestamp=curr.timestamp,
            )
        ]

    drop_rate = (prev_pct - curr_pct) / dt_min
    drop_threshold = cfg.threshold("max_battery_drop_pct_per_min", cfg.max_battery_drop_pct_per_min)
    if drop_rate > drop_threshold:
        return [
            Violation(
                "battery.impossible_drop",
                3,
                f"battery fell at {drop_rate:.1f}% per minute",
                observed=drop_rate,
                threshold=drop_threshold,
                timestamp=curr.timestamp,
            )
        ]

    low_current_threshold = cfg.threshold("low_current_threshold_a", cfg.low_current_threshold_a)
    low_current_drop_threshold = cfg.threshold(
        "max_low_current_battery_drop_pct_per_min",
        cfg.max_low_current_battery_drop_pct_per_min,
    )
    if (
        curr.battery.current_a is not None
        and abs(curr.battery.current_a) <= low_current_threshold
        and drop_rate > low_current_drop_threshold
    ):
        return [
            Violation(
                "battery.current_drop_mismatch",
                2,
                f"battery fell at {drop_rate:.1f}%/min while current draw is only {curr.battery.current_a:.1f} A",
                observed=drop_rate,
                threshold=low_current_drop_threshold,
                timestamp=curr.timestamp,
            )
        ]

    rise_rate = (curr_pct - prev_pct) / dt_min
    rise_threshold = cfg.threshold("max_battery_rise_pct_per_min", cfg.max_battery_rise_pct_per_min)
    if rise_rate > rise_threshold:
        return [
            Violation(
                "battery.impossible_rise",
                2,
                f"battery rose at {rise_rate:.1f}% per minute while in flight",
                observed=rise_rate,
                threshold=rise_threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_heading_velocity(curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    if curr.velocity is None or curr.attitude is None:
        return []

    speed = curr.velocity.horizontal_mps
    min_speed = cfg.threshold("min_heading_check_speed_mps", cfg.min_heading_check_speed_mps)
    if speed < min_speed:
        return []

    velocity_heading_deg = (degrees(atan2(curr.velocity.vy_mps, curr.velocity.vx_mps)) + 360.0) % 360.0
    residual = wrap_angle_delta_deg(curr.attitude.yaw_deg, velocity_heading_deg)
    threshold = cfg.threshold("heading_velocity_tolerance_deg", cfg.heading_velocity_tolerance_deg)
    if residual > threshold:
        return [
            Violation(
                "compass.heading_velocity_mismatch",
                2,
                f"yaw is {residual:.1f} deg away from direction of travel",
                observed=residual,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_attitude_rate(prev: DroneState, curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    if prev.attitude is None or curr.attitude is None:
        return []

    bad_time = _bad_timestamp(prev, curr)
    if bad_time:
        return [bad_time]

    dt_s = max(_dt(prev, curr), cfg.min_dt_s)
    violations: list[Violation] = []

    roll_rate = abs(curr.attitude.roll_deg - prev.attitude.roll_deg) / dt_s
    pitch_rate = abs(curr.attitude.pitch_deg - prev.attitude.pitch_deg) / dt_s
    yaw_rate = wrap_angle_delta_deg(prev.attitude.yaw_deg, curr.attitude.yaw_deg) / dt_s

    rp_threshold = cfg.threshold("max_roll_pitch_rate_deg_s", cfg.max_roll_pitch_rate_deg_s)
    yaw_threshold = cfg.threshold("max_yaw_rate_deg_s", cfg.max_yaw_rate_deg_s)

    if roll_rate > rp_threshold:
        violations.append(
            Violation("attitude.roll_rate", 2, f"roll changed at {roll_rate:.1f} deg/s", roll_rate, rp_threshold, curr.timestamp)
        )
    if pitch_rate > rp_threshold:
        violations.append(
            Violation("attitude.pitch_rate", 2, f"pitch changed at {pitch_rate:.1f} deg/s", pitch_rate, rp_threshold, curr.timestamp)
        )
    if yaw_rate > yaw_threshold:
        violations.append(
            Violation("attitude.yaw_rate", 2, f"yaw changed at {yaw_rate:.1f} deg/s", yaw_rate, yaw_threshold, curr.timestamp)
        )
    return violations


def check_sensor_freshness(curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    threshold = cfg.threshold("max_sensor_age_s", cfg.max_sensor_age_s)
    skew_threshold = cfg.threshold("max_clock_skew_s", cfg.max_clock_skew_s)
    violations: list[Violation] = []

    for sensor_name, sensor_timestamp in curr.sensor_timestamps.items():
        age = curr.timestamp - sensor_timestamp
        if age > threshold:
            violations.append(
                Violation(
                    "sensor.stale",
                    2,
                    f"{sensor_name} data is stale by {age:.2f} s",
                    observed=age,
                    threshold=threshold,
                    timestamp=curr.timestamp,
                )
            )
        elif age < -skew_threshold:
            violations.append(
                Violation(
                    "sensor.future_timestamp",
                    2,
                    f"{sensor_name} timestamp is {-age:.2f} s in the future",
                    observed=age,
                    threshold=f">= -{skew_threshold}",
                    timestamp=curr.timestamp,
                )
            )
    return violations


def check_replay_or_duplicate(states: list[DroneState], cfg: InvariantConfig) -> list[Violation]:
    curr = states[-1]
    if curr.packet_id is None:
        return []

    previous_packet_ids = {state.packet_id for state in states[:-1] if state.packet_id is not None}
    if curr.packet_id in previous_packet_ids:
        return [
            Violation(
                "packet.replay",
                3,
                f"packet id {curr.packet_id!r} was already observed",
                observed=str(curr.packet_id),
                threshold="unique packet id",
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_geofence(curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    violations: list[Violation] = []
    violations.extend(_point_geofence_violations("state", curr.position, curr.timestamp, cfg))

    if curr.last_command is not None and curr.last_command.target is not None:
        violations.extend(_point_geofence_violations("command", curr.last_command.target, curr.timestamp, cfg))

    return violations


def _point_geofence_violations(label: str, point, timestamp: float, cfg: InvariantConfig) -> list[Violation]:
    violations: list[Violation] = []
    checks = (
        ("lat", point.lat, cfg.geofence_min_lat, cfg.geofence_max_lat),
        ("lon", point.lon, cfg.geofence_min_lon, cfg.geofence_max_lon),
        ("alt_m", point.alt_m, cfg.geofence_min_alt_m, cfg.geofence_max_alt_m),
    )
    for axis, value, lower, upper in checks:
        if lower is not None and value < lower:
            violations.append(
                Violation(
                    f"geofence.{label}_{axis}",
                    3,
                    f"{label} {axis}={value:.6f} is below geofence minimum",
                    observed=value,
                    threshold=lower,
                    timestamp=timestamp,
                )
            )
        if upper is not None and value > upper:
            violations.append(
                Violation(
                    f"geofence.{label}_{axis}",
                    3,
                    f"{label} {axis}={value:.6f} is above geofence maximum",
                    observed=value,
                    threshold=upper,
                    timestamp=timestamp,
                )
            )
    return violations


def check_command_plausibility(curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    command = curr.last_command
    if command is None or command.target is None:
        return []

    violations: list[Violation] = []
    distance = distance_3d_m(curr.position, command.target)
    max_distance = cfg.threshold("max_command_distance_m", cfg.max_command_distance_m)
    if distance > max_distance:
        violations.append(
            Violation(
                "command.distance",
                2,
                f"{command.command_type} target is {distance:.1f} m away",
                observed=distance,
                threshold=max_distance,
                timestamp=curr.timestamp,
            )
        )

    deadline_s = command.metadata.get("deadline_s", command.metadata.get("eta_s"))
    if deadline_s is not None:
        try:
            deadline = float(deadline_s)
        except (TypeError, ValueError):
            deadline = 0.0
        if deadline <= 0:
            violations.append(
                Violation(
                    "command.deadline",
                    2,
                    "command deadline must be positive",
                    observed=str(deadline_s),
                    threshold="> 0",
                    timestamp=curr.timestamp,
                )
            )
        else:
            required_ground_speed = horizontal_distance_m(curr.position, command.target) / deadline
            required_vertical_speed = abs(command.target.alt_m - curr.position.alt_m) / deadline
            if required_ground_speed > cfg.max_ground_speed_mps:
                violations.append(
                    Violation(
                        "command.required_ground_speed",
                        3,
                        f"command requires {required_ground_speed:.1f} m/s ground speed",
                        observed=required_ground_speed,
                        threshold=cfg.max_ground_speed_mps,
                        timestamp=curr.timestamp,
                    )
                )
            if required_vertical_speed > cfg.max_vertical_speed_mps:
                violations.append(
                    Violation(
                        "command.required_vertical_speed",
                        3,
                        f"command requires {required_vertical_speed:.1f} m/s vertical speed",
                        observed=required_vertical_speed,
                        threshold=cfg.max_vertical_speed_mps,
                        timestamp=curr.timestamp,
                    )
                )

            if curr.attitude is not None:
                target_heading = (
                    degrees(
                        atan2(
                            command.target.lon - curr.position.lon,
                            command.target.lat - curr.position.lat,
                        )
                    )
                    + 360.0
                ) % 360.0
                required_yaw_rate = wrap_angle_delta_deg(curr.attitude.yaw_deg, target_heading) / deadline
                if required_yaw_rate > cfg.max_yaw_rate_deg_s:
                    violations.append(
                        Violation(
                            "command.required_yaw_rate",
                            3,
                            f"command requires {required_yaw_rate:.1f} deg/s yaw rate",
                            observed=required_yaw_rate,
                            threshold=cfg.max_yaw_rate_deg_s,
                            timestamp=curr.timestamp,
                        )
                    )
    return violations


def check_ground_altitude(curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    if curr.ground_distance_m is None:
        return []

    expected_ground_distance = curr.position.alt_m - cfg.terrain_alt_m
    residual = abs(curr.ground_distance_m - expected_ground_distance)
    threshold = cfg.threshold("max_ground_altitude_residual_m", cfg.max_ground_altitude_residual_m)
    if residual > threshold:
        return [
            Violation(
                "altitude.ground_range_mismatch",
                2,
                f"altitude implies {expected_ground_distance:.1f} m AGL but rangefinder reads {curr.ground_distance_m:.1f} m",
                observed=residual,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_multi_drone_consistency(curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    violations: list[Violation] = []
    for peer in curr.peer_states:
        age = curr.timestamp - peer.timestamp
        if age > cfg.max_peer_state_age_s:
            violations.append(
                Violation(
                    "peer.stale",
                    2,
                    f"peer {peer.drone_id} state is stale by {age:.2f} s",
                    observed=age,
                    threshold=cfg.max_peer_state_age_s,
                    timestamp=curr.timestamp,
                )
            )

        if peer.drone_id == curr.drone_id:
            violations.append(
                Violation(
                    "peer.duplicate_identity",
                    3,
                    f"peer state reuses local drone id {curr.drone_id!r}",
                    observed=peer.drone_id,
                    threshold="distinct drone id",
                    timestamp=curr.timestamp,
                )
            )

        if peer.reported_waypoint_reached and peer.current_waypoint is not None:
            peer_dist = distance_3d_m(peer.position, peer.current_waypoint)
            if peer_dist > cfg.waypoint_radius_m:
                violations.append(
                    Violation(
                        "peer.false_waypoint_reached",
                        3,
                        f"peer {peer.drone_id} reports waypoint reached but is {peer_dist:.1f} m away",
                        observed=peer_dist,
                        threshold=cfg.waypoint_radius_m,
                        timestamp=curr.timestamp,
                    )
                )

        peer_distance = distance_3d_m(curr.position, peer.position)
        if peer_distance < cfg.min_peer_separation_m:
            curr_wp = curr.current_waypoint
            peer_wp = peer.current_waypoint
            waypoints_are_distinct = (
                curr_wp is not None
                and peer_wp is not None
                and distance_3d_m(curr_wp, peer_wp) > cfg.waypoint_radius_m
            )
            if waypoints_are_distinct:
                violations.append(
                    Violation(
                        "peer.position_plan_conflict",
                        2,
                        f"{curr.drone_id} and {peer.drone_id} report nearly same position despite distinct waypoints",
                        observed=peer_distance,
                        threshold=cfg.min_peer_separation_m,
                        timestamp=curr.timestamp,
                    )
                )
    return violations


def check_perception_depth(curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    threshold = cfg.threshold("perception_depth_tolerance_m", cfg.perception_depth_tolerance_m)
    violations: list[Violation] = []
    for obj in curr.perception_objects:
        if obj.claimed_range_m is None or obj.depth_range_m is None:
            continue

        residual = abs(obj.claimed_range_m - obj.depth_range_m)
        if residual > threshold:
            violations.append(
                Violation(
                    "perception.depth_mismatch",
                    3,
                    f"{obj.class_id} claims {obj.claimed_range_m:.1f} m but depth reads {obj.depth_range_m:.1f} m",
                    observed=residual,
                    threshold=threshold,
                    timestamp=curr.timestamp,
                )
            )
    return violations


def check_waypoint_claim(curr: DroneState, cfg: InvariantConfig) -> list[Violation]:
    if not curr.reported_waypoint_reached or curr.current_waypoint is None:
        return []

    dist = distance_3d_m(curr.position, curr.current_waypoint)
    threshold = cfg.threshold("waypoint_radius_m", cfg.waypoint_radius_m)
    if dist > threshold:
        return [
            Violation(
                "mission.false_waypoint_reached",
                3,
                f"waypoint reported reached but position is {dist:.1f} m away",
                observed=dist,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []


def check_command_effect(states: list[DroneState], cfg: InvariantConfig) -> list[Violation]:
    curr = states[-1]
    command = curr.last_command
    if command is None or command.command_type.lower() not in {"goto", "avoid"} or command.target is None:
        return []

    elapsed = curr.timestamp - command.issued_at
    if elapsed < cfg.command_progress_window_s:
        return []

    since_command = [s for s in states if s.timestamp >= command.issued_at]
    if len(since_command) < 2:
        return []

    start = since_command[0]
    start_dist = distance_3d_m(start.position, command.target)
    curr_dist = distance_3d_m(curr.position, command.target)
    progress = start_dist - curr_dist
    threshold = cfg.threshold("command_min_progress_m", cfg.command_min_progress_m)

    if progress < threshold:
        return [
            Violation(
                "command.no_effect",
                2,
                f"{command.command_type} command has not reduced target distance after {elapsed:.1f} s",
                observed=progress,
                threshold=threshold,
                timestamp=curr.timestamp,
            )
        ]
    return []

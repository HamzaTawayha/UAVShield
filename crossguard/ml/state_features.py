from __future__ import annotations

import json
from dataclasses import dataclass
from math import cos, radians
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

from crossguard.defense.state import DroneState, GeoPoint
from crossguard.utils.geo import EARTH_RADIUS_M, distance_3d_m, horizontal_distance_m


RUNTIME_FEATURE_NAMES = [
    "position.local_x_m",
    "position.local_y_m",
    "position.alt_m",
    "velocity.vx_mps",
    "velocity.vy_mps",
    "velocity.vz_mps",
    "velocity.horizontal_mps",
    "velocity.magnitude_mps",
    "acceleration.ax_mps2",
    "acceleration.ay_mps2",
    "acceleration.az_mps2",
    "acceleration.magnitude_mps2",
    "attitude.roll_deg",
    "attitude.pitch_deg",
    "attitude.yaw_deg",
    "battery.percent",
    "battery.current_a",
    "battery.voltage_v",
    "ground_distance_m",
    "sensor.max_age_s",
    "sensor.mean_age_s",
    "sensor.count",
    "command.target_distance_m",
    "command.required_ground_speed_mps",
    "command.required_vertical_speed_mps",
    "waypoint.distance_m",
    "waypoint.reached_claim",
    "perception.min_claimed_range_m",
    "perception.min_depth_range_m",
    "perception.max_range_residual_m",
    "peer.count",
    "peer.min_separation_m",
]


@dataclass(frozen=True)
class WindowNormalization:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    source: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "WindowNormalization":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = tuple(str(name) for name in payload["feature_names"])
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        if mean.shape[0] != len(feature_names) or std.shape[0] != len(feature_names):
            raise ValueError(f"{path} has inconsistent feature_names/mean/std lengths")
        std = np.where(std < 1e-6, 1.0, std)
        return cls(
            feature_names=feature_names,
            mean=mean,
            std=std,
            source=str(payload.get("source", "")),
        )

    def normalize(self, raw_window: np.ndarray) -> np.ndarray:
        """Normalize a raw window and treat missing features as neutral evidence."""

        if raw_window.shape[0] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} feature rows, got {raw_window.shape[0]}"
            )
        normalized = (raw_window - self.mean[:, None]) / self.std[:, None]
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def state_window_to_array(
    states: Iterable[DroneState],
    feature_names: Iterable[str] = RUNTIME_FEATURE_NAMES,
) -> np.ndarray:
    """Convert DroneState history into a feature x time matrix.

    Missing runtime fields are encoded as NaN so normalization can map them to
    neutral zero-valued evidence.
    """

    state_list = list(states)
    names = tuple(feature_names)
    if not state_list:
        return np.empty((len(names), 0), dtype=np.float32)

    origin = state_list[0].position
    rows = []
    for state in state_list:
        features = features_for_state(state, origin)
        rows.append([features.get(name, float("nan")) for name in names])
    return np.asarray(rows, dtype=np.float32).T


def features_for_state(state: DroneState, origin: GeoPoint | None = None) -> dict[str, float]:
    origin = origin or state.position
    local_x, local_y = _local_xy_m(origin, state.position)
    features: dict[str, float] = {
        "position.local_x_m": local_x,
        "position.local_y_m": local_y,
        "position.alt_m": float(state.position.alt_m),
        "sensor.count": float(len(state.sensor_timestamps)),
        "waypoint.reached_claim": 1.0 if state.reported_waypoint_reached else 0.0,
        "peer.count": float(len(state.peer_states)),
    }

    if state.velocity is not None:
        features.update(
            {
                "velocity.vx_mps": float(state.velocity.vx_mps),
                "velocity.vy_mps": float(state.velocity.vy_mps),
                "velocity.vz_mps": float(state.velocity.vz_mps),
                "velocity.horizontal_mps": float(state.velocity.horizontal_mps),
                "velocity.magnitude_mps": float(state.velocity.magnitude_mps),
            }
        )

    if state.acceleration is not None:
        features.update(
            {
                "acceleration.ax_mps2": float(state.acceleration.ax_mps2),
                "acceleration.ay_mps2": float(state.acceleration.ay_mps2),
                "acceleration.az_mps2": float(state.acceleration.az_mps2),
                "acceleration.magnitude_mps2": float(state.acceleration.magnitude_mps2),
            }
        )

    if state.attitude is not None:
        features.update(
            {
                "attitude.roll_deg": float(state.attitude.roll_deg),
                "attitude.pitch_deg": float(state.attitude.pitch_deg),
                "attitude.yaw_deg": float(state.attitude.yaw_deg),
            }
        )

    if state.battery is not None:
        features["battery.percent"] = float(state.battery.percent)
        if state.battery.current_a is not None:
            features["battery.current_a"] = float(state.battery.current_a)
        if state.battery.voltage_v is not None:
            features["battery.voltage_v"] = float(state.battery.voltage_v)

    if state.ground_distance_m is not None:
        features["ground_distance_m"] = float(state.ground_distance_m)

    sensor_ages = [state.timestamp - ts for ts in state.sensor_timestamps.values()]
    if sensor_ages:
        features["sensor.max_age_s"] = float(max(sensor_ages))
        features["sensor.mean_age_s"] = float(mean(sensor_ages))

    if state.last_command is not None and state.last_command.target is not None:
        target = state.last_command.target
        dt_s = max(state.timestamp - state.last_command.issued_at, 1e-6)
        horizontal = horizontal_distance_m(state.position, target)
        vertical = abs(target.alt_m - state.position.alt_m)
        features["command.target_distance_m"] = float(distance_3d_m(state.position, target))
        features["command.required_ground_speed_mps"] = float(horizontal / dt_s)
        features["command.required_vertical_speed_mps"] = float(vertical / dt_s)

    if state.current_waypoint is not None:
        features["waypoint.distance_m"] = float(distance_3d_m(state.position, state.current_waypoint))

    claimed = [
        obj.claimed_range_m
        for obj in state.perception_objects
        if obj.claimed_range_m is not None
    ]
    depth = [
        obj.depth_range_m
        for obj in state.perception_objects
        if obj.depth_range_m is not None
    ]
    residuals = [
        abs(obj.claimed_range_m - obj.depth_range_m)
        for obj in state.perception_objects
        if obj.claimed_range_m is not None and obj.depth_range_m is not None
    ]
    if claimed:
        features["perception.min_claimed_range_m"] = float(min(claimed))
    if depth:
        features["perception.min_depth_range_m"] = float(min(depth))
    if residuals:
        features["perception.max_range_residual_m"] = float(max(residuals))

    separations = [distance_3d_m(state.position, peer.position) for peer in state.peer_states]
    if separations:
        features["peer.min_separation_m"] = float(min(separations))

    for name, value in state.extra_features.items():
        try:
            features[name] = float(value)
        except (TypeError, ValueError):
            continue

    return features


def _local_xy_m(origin: GeoPoint, point: GeoPoint) -> tuple[float, float]:
    dlat = radians(point.lat - origin.lat)
    dlon = radians(point.lon - origin.lon)
    lat = radians((point.lat + origin.lat) / 2.0)
    x_north = dlat * EARTH_RADIUS_M
    y_east = dlon * EARTH_RADIUS_M * cos(lat)
    return float(x_north), float(y_east)

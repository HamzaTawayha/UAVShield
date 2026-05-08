from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeoPoint:
    """Geodetic position used by telemetry and command targets."""

    lat: float
    lon: float
    alt_m: float = 0.0


@dataclass(frozen=True)
class Velocity:
    """Velocity in a local NED-like frame, meters per second."""

    vx_mps: float
    vy_mps: float
    vz_mps: float = 0.0

    @property
    def horizontal_mps(self) -> float:
        return (self.vx_mps**2 + self.vy_mps**2) ** 0.5

    @property
    def magnitude_mps(self) -> float:
        return (self.vx_mps**2 + self.vy_mps**2 + self.vz_mps**2) ** 0.5


@dataclass(frozen=True)
class Acceleration:
    """IMU acceleration in meters per second squared."""

    ax_mps2: float
    ay_mps2: float
    az_mps2: float = 0.0

    @property
    def magnitude_mps2(self) -> float:
        return (self.ax_mps2**2 + self.ay_mps2**2 + self.az_mps2**2) ** 0.5


@dataclass(frozen=True)
class Attitude:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass(frozen=True)
class BatteryState:
    percent: float
    voltage_v: float | None = None
    current_a: float | None = None


@dataclass(frozen=True)
class PerceptionObject:
    """Perception summary with an independently checkable range claim."""

    class_id: str
    bbox_center_x: float
    bbox_center_y: float
    confidence: float
    claimed_range_m: float | None = None
    depth_range_m: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None


@dataclass(frozen=True)
class Command:
    """Planner command observed by the monitor."""

    command_type: str
    issued_at: float
    target: GeoPoint | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeerDroneState:
    """Minimal shared state from another drone or coordinator."""

    drone_id: str
    timestamp: float
    position: GeoPoint
    current_waypoint: GeoPoint | None = None
    reported_waypoint_reached: bool = False


@dataclass(frozen=True)
class DroneState:
    """Single normalized snapshot consumed by the sanity harness."""

    timestamp: float
    position: GeoPoint
    drone_id: str = "drone1"
    packet_id: str | None = None
    velocity: Velocity | None = None
    acceleration: Acceleration | None = None
    attitude: Attitude | None = None
    battery: BatteryState | None = None
    ground_distance_m: float | None = None
    sensor_timestamps: dict[str, float] = field(default_factory=dict)
    perception_objects: tuple[PerceptionObject, ...] = ()
    peer_states: tuple[PeerDroneState, ...] = ()
    last_command: Command | None = None
    current_waypoint: GeoPoint | None = None
    reported_waypoint_reached: bool = False
    source: str = "unknown"


@dataclass(frozen=True)
class Violation:
    check_id: str
    severity: int
    message: str
    observed: float | str | None = None
    threshold: float | str | None = None
    timestamp: float | None = None

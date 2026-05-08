from __future__ import annotations

import json
from math import asin, atan2, degrees
from pathlib import Path

from crossguard.defense.config_loader import load_invariant_config
from crossguard.defense.harness import CrossGuardHarness
from crossguard.defense.invariants import InvariantConfig
from crossguard.defense.state import Acceleration, Attitude, BatteryState, Command, DroneState, GeoPoint, Velocity

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from rclpy.node import Node
    from sensor_msgs.msg import BatteryState as RosBatteryState
    from sensor_msgs.msg import Imu, Range
    from std_msgs.msg import Bool, String
except ModuleNotFoundError:  # pragma: no cover - exercised only on ROS2 machines.
    rclpy = None
    Node = object
    PoseStamped = object
    TwistStamped = object
    Imu = object
    Range = object
    RosBatteryState = object
    Bool = object
    String = object


METERS_PER_DEGREE = 111_139.0


def local_pose_to_geopoint(msg: PoseStamped) -> GeoPoint:
    """Map local simulator meters into tiny geodetic deltas for invariant math."""

    return GeoPoint(
        lat=msg.pose.position.x / METERS_PER_DEGREE,
        lon=msg.pose.position.y / METERS_PER_DEGREE,
        alt_m=msg.pose.position.z,
    )


def _quaternion_to_attitude(x: float, y: float, z: float, w: float) -> Attitude:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = degrees(atan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = 90.0 if sinp > 0 else -90.0
    else:
        pitch = degrees(asin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = (degrees(atan2(siny_cosp, cosy_cosp)) + 360.0) % 360.0
    return Attitude(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)


class CrossGuardRosNode(Node):
    """ROS2 adapter around the pure-Python CrossGuardHarness."""

    def __init__(self, config: InvariantConfig | None = None) -> None:
        super().__init__("crossguard")
        self.declare_parameter("position_topic", "/drone1/telemetry/position")
        self.declare_parameter("velocity_topic", "/drone1/telemetry/velocity")
        self.declare_parameter("battery_topic", "/drone1/telemetry/battery")
        self.declare_parameter("imu_topic", "/drone1/imu")
        self.declare_parameter("range_topic", "/drone1/rangefinder")
        self.declare_parameter("command_topic", "/drone1/agent/goto_target")
        self.declare_parameter("alert_topic", "/crossguard/alert")
        self.declare_parameter("violations_topic", "/crossguard/violations")
        self.declare_parameter("hover_request_topic", "/crossguard/hover_request")
        self.declare_parameter("history_size", 300)
        self.declare_parameter("alert_threshold", 3)

        history_size = int(self.get_parameter("history_size").value)
        alert_threshold = int(self.get_parameter("alert_threshold").value)
        self.harness = CrossGuardHarness(
            config=config or InvariantConfig(),
            history_size=history_size,
            alert_threshold=alert_threshold,
        )

        self.position: GeoPoint | None = None
        self.velocity: Velocity | None = None
        self.acceleration: Acceleration | None = None
        self.attitude: Attitude | None = None
        self.battery: BatteryState | None = None
        self.ground_distance_m: float | None = None
        self.last_command: Command | None = None
        self.sensor_timestamps: dict[str, float] = {}

        self.create_subscription(
            PoseStamped,
            self.get_parameter("position_topic").value,
            self.position_cb,
            10,
        )
        self.create_subscription(
            TwistStamped,
            self.get_parameter("velocity_topic").value,
            self.velocity_cb,
            10,
        )
        self.create_subscription(
            RosBatteryState,
            self.get_parameter("battery_topic").value,
            self.battery_cb,
            10,
        )
        self.create_subscription(
            Imu,
            self.get_parameter("imu_topic").value,
            self.imu_cb,
            10,
        )
        self.create_subscription(
            Range,
            self.get_parameter("range_topic").value,
            self.range_cb,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("command_topic").value,
            self.command_cb,
            10,
        )

        self.alert_pub = self.create_publisher(Bool, self.get_parameter("alert_topic").value, 10)
        self.hover_pub = self.create_publisher(Bool, self.get_parameter("hover_request_topic").value, 10)
        self.violations_pub = self.create_publisher(String, self.get_parameter("violations_topic").value, 10)
        self.timer = self.create_timer(0.1, self.monitor_cycle)

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def stamp_seconds(self, msg) -> float:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return self.now_seconds()
        value = float(stamp.sec) + float(stamp.nanosec) / 1e9
        return value if value > 0 else self.now_seconds()

    def position_cb(self, msg: PoseStamped) -> None:
        self.position = local_pose_to_geopoint(msg)
        self.sensor_timestamps["position"] = self.stamp_seconds(msg)

    def velocity_cb(self, msg: TwistStamped) -> None:
        self.velocity = Velocity(
            vx_mps=msg.twist.linear.x,
            vy_mps=msg.twist.linear.y,
            vz_mps=msg.twist.linear.z,
        )
        self.sensor_timestamps["velocity"] = self.stamp_seconds(msg)

    def battery_cb(self, msg: RosBatteryState) -> None:
        pct = float(msg.percentage)
        if pct <= 1.0:
            pct *= 100.0
        self.battery = BatteryState(
            percent=pct,
            voltage_v=float(msg.voltage) if msg.voltage == msg.voltage else None,
            current_a=float(msg.current) if msg.current == msg.current else None,
        )
        self.sensor_timestamps["battery"] = self.stamp_seconds(msg)

    def imu_cb(self, msg: Imu) -> None:
        self.acceleration = Acceleration(
            ax_mps2=msg.linear_acceleration.x,
            ay_mps2=msg.linear_acceleration.y,
            az_mps2=msg.linear_acceleration.z,
        )
        self.attitude = _quaternion_to_attitude(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        self.sensor_timestamps["imu"] = self.stamp_seconds(msg)

    def range_cb(self, msg: Range) -> None:
        self.ground_distance_m = float(msg.range)
        self.sensor_timestamps["rangefinder"] = self.stamp_seconds(msg)

    def command_cb(self, msg: PoseStamped) -> None:
        self.last_command = Command(
            command_type="goto",
            issued_at=self.now_seconds(),
            target=local_pose_to_geopoint(msg),
        )

    def monitor_cycle(self) -> None:
        if self.position is None:
            return

        state = DroneState(
            timestamp=self.now_seconds(),
            position=self.position,
            velocity=self.velocity,
            acceleration=self.acceleration,
            attitude=self.attitude,
            battery=self.battery,
            ground_distance_m=self.ground_distance_m,
            sensor_timestamps=dict(self.sensor_timestamps),
            last_command=self.last_command,
            source="ros2",
        )
        decision = self.harness.observe(state)

        self.alert_pub.publish(Bool(data=decision.alert))
        if decision.alert:
            self.hover_pub.publish(Bool(data=True))

        payload = {
            "timestamp": state.timestamp,
            "alert": decision.alert,
            "suspicion": decision.suspicion,
            "violations": [
                {
                    "check_id": v.check_id,
                    "severity": v.severity,
                    "message": v.message,
                    "observed": v.observed,
                    "threshold": v.threshold,
                }
                for v in decision.violations
            ],
        }
        self.violations_pub.publish(String(data=json.dumps(payload)))


def main() -> None:
    if rclpy is None:
        raise SystemExit("ROS2/rclpy is not installed in this Python environment.")

    rclpy.init()
    default_config = Path(__file__).resolve().parents[1] / "config" / "crossguard_params.yaml"
    node = CrossGuardRosNode(config=load_invariant_config(default_config))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_TO_ID = {
    "Normal": 0,
    "Engine Failure": 1,
    "Aileron Fault": 2,
    "Elevator Fault": 3,
    "Rudder Fault": 4,
    "Multiple Control Faults": 5,
    "Unknown Fault": 6,
}

FAILURE_TOPIC_TO_CLASS = {
    "engines": "Engine Failure",
    "aileron": "Aileron Fault",
    "elevator": "Elevator Fault",
    "rudder": "Rudder Fault",
}


@dataclass(frozen=True)
class CsvChannelSpec:
    suffix: str
    column: str
    feature_name: str
    transform: str = "identity"


@dataclass(frozen=True)
class SequenceInfo:
    path: Path
    name: str
    class_name: str
    failure_start_s: float | None
    failure_topics: tuple[str, ...]


@dataclass(frozen=True)
class WindowRecord:
    x: np.ndarray
    y: int
    class_name: str
    sequence: str
    path: str
    window_start_s: float
    role: str
    failure_start_s: float | None = None
    failure_topics: str = ""


CHANNELS = [
    CsvChannelSpec("mavros-local_position-pose.csv", "field.pose.position.x", "position.local_x_m"),
    CsvChannelSpec("mavros-local_position-pose.csv", "field.pose.position.y", "position.local_y_m"),
    CsvChannelSpec("mavros-local_position-pose.csv", "field.pose.position.z", "position.local_z_m"),
    CsvChannelSpec("mavros-local_position-velocity.csv", "field.twist.linear.x", "velocity.local_x_mps"),
    CsvChannelSpec("mavros-local_position-velocity.csv", "field.twist.linear.y", "velocity.local_y_mps"),
    CsvChannelSpec("mavros-local_position-velocity.csv", "field.twist.linear.z", "velocity.local_z_mps"),
    CsvChannelSpec("mavros-local_position-velocity.csv", "field.twist.angular.x", "attitude.roll_rate_rad_s"),
    CsvChannelSpec("mavros-local_position-velocity.csv", "field.twist.angular.y", "attitude.pitch_rate_rad_s"),
    CsvChannelSpec("mavros-local_position-velocity.csv", "field.twist.angular.z", "attitude.yaw_rate_rad_s"),
    CsvChannelSpec("mavros-global_position-local.csv", "field.pose.pose.position.x", "position.global_local_x_m"),
    CsvChannelSpec("mavros-global_position-local.csv", "field.pose.pose.position.y", "position.global_local_y_m"),
    CsvChannelSpec("mavros-global_position-local.csv", "field.pose.pose.position.z", "position.global_local_z_m"),
    CsvChannelSpec("mavros-global_position-local.csv", "field.pose.covariance0", "position.local_cov_x"),
    CsvChannelSpec("mavros-global_position-local.csv", "field.pose.covariance7", "position.local_cov_y"),
    CsvChannelSpec("mavros-global_position-local.csv", "field.pose.covariance14", "position.local_cov_z"),
    CsvChannelSpec("mavros-global_position-global.csv", "field.latitude", "position.global_lat"),
    CsvChannelSpec("mavros-global_position-global.csv", "field.longitude", "position.global_lon"),
    CsvChannelSpec("mavros-global_position-global.csv", "field.altitude", "position.global_alt_m"),
    CsvChannelSpec("mavros-global_position-raw-fix.csv", "field.latitude", "position.raw_gps_lat"),
    CsvChannelSpec("mavros-global_position-raw-fix.csv", "field.longitude", "position.raw_gps_lon"),
    CsvChannelSpec("mavros-global_position-raw-fix.csv", "field.altitude", "position.raw_gps_alt_m"),
    CsvChannelSpec("mavros-global_position-raw-gps_vel.csv", "field.twist.linear.x", "velocity.gps_x_mps"),
    CsvChannelSpec("mavros-global_position-raw-gps_vel.csv", "field.twist.linear.y", "velocity.gps_y_mps"),
    CsvChannelSpec("mavros-global_position-raw-gps_vel.csv", "field.twist.linear.z", "velocity.gps_z_mps"),
    CsvChannelSpec("mavros-global_position-compass_hdg.csv", "field.data", "position.heading_deg"),
    CsvChannelSpec("mavros-global_position-rel_alt.csv", "field.data", "position.relative_alt_m"),
    CsvChannelSpec("mavros-imu-data.csv", "field.orientation.x", "attitude.qx"),
    CsvChannelSpec("mavros-imu-data.csv", "field.orientation.y", "attitude.qy"),
    CsvChannelSpec("mavros-imu-data.csv", "field.orientation.z", "attitude.qz"),
    CsvChannelSpec("mavros-imu-data.csv", "field.orientation.w", "attitude.qw"),
    CsvChannelSpec("mavros-imu-data.csv", "field.angular_velocity.x", "sensor.gyro_x_rad_s"),
    CsvChannelSpec("mavros-imu-data.csv", "field.angular_velocity.y", "sensor.gyro_y_rad_s"),
    CsvChannelSpec("mavros-imu-data.csv", "field.angular_velocity.z", "sensor.gyro_z_rad_s"),
    CsvChannelSpec("mavros-imu-data.csv", "field.linear_acceleration.x", "acceleration.ax_mps2"),
    CsvChannelSpec("mavros-imu-data.csv", "field.linear_acceleration.y", "acceleration.ay_mps2"),
    CsvChannelSpec("mavros-imu-data.csv", "field.linear_acceleration.z", "acceleration.az_mps2"),
    CsvChannelSpec("mavros-imu-data_raw.csv", "field.angular_velocity.x", "sensor.raw_gyro_x_rad_s"),
    CsvChannelSpec("mavros-imu-data_raw.csv", "field.angular_velocity.y", "sensor.raw_gyro_y_rad_s"),
    CsvChannelSpec("mavros-imu-data_raw.csv", "field.angular_velocity.z", "sensor.raw_gyro_z_rad_s"),
    CsvChannelSpec("mavros-imu-data_raw.csv", "field.linear_acceleration.x", "acceleration.raw_ax_mps2"),
    CsvChannelSpec("mavros-imu-data_raw.csv", "field.linear_acceleration.y", "acceleration.raw_ay_mps2"),
    CsvChannelSpec("mavros-imu-data_raw.csv", "field.linear_acceleration.z", "acceleration.raw_az_mps2"),
    CsvChannelSpec("mavros-imu-mag.csv", "field.magnetic_field.x", "sensor.mag_x"),
    CsvChannelSpec("mavros-imu-mag.csv", "field.magnetic_field.y", "sensor.mag_y"),
    CsvChannelSpec("mavros-imu-mag.csv", "field.magnetic_field.z", "sensor.mag_z"),
    CsvChannelSpec("mavros-imu-atm_pressure.csv", "field.fluid_pressure", "sensor.atm_pressure_pa"),
    CsvChannelSpec("mavros-imu-temperature.csv", "field.temperature", "sensor.imu_temperature_c"),
    CsvChannelSpec("mavros-battery.csv", "field.voltage", "battery.voltage_v"),
    CsvChannelSpec("mavros-battery.csv", "field.current", "battery.current_a"),
    CsvChannelSpec("mavros-battery.csv", "field.charge", "battery.charge_ah"),
    CsvChannelSpec("mavros-battery.csv", "field.capacity", "battery.capacity_ah"),
    CsvChannelSpec("mavros-battery.csv", "field.percentage", "battery.percent", "percent"),
    CsvChannelSpec("mavros-battery.csv", "field.power_supply_status", "battery.power_supply_status"),
    CsvChannelSpec("mavros-battery.csv", "field.power_supply_health", "battery.power_supply_health"),
    CsvChannelSpec("mavros-nav_info-airspeed.csv", "field.commanded", "nav.airspeed_commanded_mps"),
    CsvChannelSpec("mavros-nav_info-airspeed.csv", "field.measured", "nav.airspeed_measured_mps"),
    CsvChannelSpec("mavros-nav_info-errors.csv", "field.alt_error", "nav.alt_error_m"),
    CsvChannelSpec("mavros-nav_info-errors.csv", "field.aspd_error", "nav.airspeed_error_mps"),
    CsvChannelSpec("mavros-nav_info-errors.csv", "field.xtrack_error", "nav.xtrack_error_m"),
    CsvChannelSpec("mavros-nav_info-errors.csv", "field.wp_dist", "nav.waypoint_distance_m"),
    CsvChannelSpec("mavros-nav_info-roll.csv", "field.commanded", "nav.roll_commanded_deg"),
    CsvChannelSpec("mavros-nav_info-roll.csv", "field.measured", "nav.roll_measured_deg"),
    CsvChannelSpec("mavros-nav_info-pitch.csv", "field.commanded", "nav.pitch_commanded_deg"),
    CsvChannelSpec("mavros-nav_info-pitch.csv", "field.measured", "nav.pitch_measured_deg"),
    CsvChannelSpec("mavros-nav_info-yaw.csv", "field.commanded", "nav.yaw_commanded_deg"),
    CsvChannelSpec("mavros-nav_info-yaw.csv", "field.measured", "nav.yaw_measured_deg"),
    CsvChannelSpec("mavros-nav_info-velocity.csv", "field.commanded", "nav.velocity_commanded_mps"),
    CsvChannelSpec("mavros-nav_info-velocity.csv", "field.measured", "nav.velocity_measured_mps"),
    CsvChannelSpec("mavros-vfr_hud.csv", "field.airspeed", "vfr.airspeed_mps"),
    CsvChannelSpec("mavros-vfr_hud.csv", "field.groundspeed", "vfr.groundspeed_mps"),
    CsvChannelSpec("mavros-vfr_hud.csv", "field.heading", "vfr.heading_deg"),
    CsvChannelSpec("mavros-vfr_hud.csv", "field.throttle", "vfr.throttle"),
    CsvChannelSpec("mavros-vfr_hud.csv", "field.altitude", "vfr.altitude_m"),
    CsvChannelSpec("mavros-vfr_hud.csv", "field.climb", "vfr.climb_mps"),
    CsvChannelSpec("mavros-wind_estimation.csv", "field.twist.linear.x", "wind.x_mps"),
    CsvChannelSpec("mavros-wind_estimation.csv", "field.twist.linear.y", "wind.y_mps"),
    CsvChannelSpec("mavros-wind_estimation.csv", "field.twist.linear.z", "wind.z_mps"),
    CsvChannelSpec("mavctrl-path_dev.csv", "field.x", "control.path_dev_x_m"),
    CsvChannelSpec("mavctrl-path_dev.csv", "field.y", "control.path_dev_y_m"),
    CsvChannelSpec("mavctrl-path_dev.csv", "field.z", "control.path_dev_z_m"),
    CsvChannelSpec("mavctrl-rpy.csv", "field.x", "control.roll_command_rad"),
    CsvChannelSpec("mavctrl-rpy.csv", "field.y", "control.pitch_command_rad"),
    CsvChannelSpec("mavctrl-rpy.csv", "field.z", "control.yaw_command_rad"),
    CsvChannelSpec("mavros-rc-in.csv", "field.rssi", "telemetry.rc_rssi"),
    CsvChannelSpec("mavros-rc-in.csv", "field.channels0", "rc.input_0"),
    CsvChannelSpec("mavros-rc-in.csv", "field.channels1", "rc.input_1"),
    CsvChannelSpec("mavros-rc-in.csv", "field.channels2", "rc.input_2"),
    CsvChannelSpec("mavros-rc-in.csv", "field.channels3", "rc.input_3"),
    CsvChannelSpec("mavros-rc-out.csv", "field.channels0", "actuator.output_0"),
    CsvChannelSpec("mavros-rc-out.csv", "field.channels1", "actuator.output_1"),
    CsvChannelSpec("mavros-rc-out.csv", "field.channels2", "actuator.output_2"),
    CsvChannelSpec("mavros-rc-out.csv", "field.channels3", "actuator.output_3"),
    CsvChannelSpec("mavros-rc-out.csv", "field.channels4", "actuator.output_4"),
    CsvChannelSpec("mavros-rc-out.csv", "field.channels5", "actuator.output_5"),
    CsvChannelSpec("mavros-setpoint_raw-local.csv", "field.position.x", "command.local_x_m"),
    CsvChannelSpec("mavros-setpoint_raw-local.csv", "field.position.y", "command.local_y_m"),
    CsvChannelSpec("mavros-setpoint_raw-local.csv", "field.position.z", "command.local_z_m"),
    CsvChannelSpec("mavros-setpoint_raw-local.csv", "field.velocity.x", "command.local_vx_mps"),
    CsvChannelSpec("mavros-setpoint_raw-local.csv", "field.velocity.y", "command.local_vy_mps"),
    CsvChannelSpec("mavros-setpoint_raw-local.csv", "field.velocity.z", "command.local_vz_mps"),
    CsvChannelSpec("mavros-setpoint_raw-target_global.csv", "field.latitude", "command.global_lat"),
    CsvChannelSpec("mavros-setpoint_raw-target_global.csv", "field.longitude", "command.global_lon"),
    CsvChannelSpec("mavros-setpoint_raw-target_global.csv", "field.altitude", "command.global_alt_m"),
    CsvChannelSpec("mavros-state.csv", "field.connected", "state.connected"),
    CsvChannelSpec("mavros-state.csv", "field.armed", "state.armed"),
    CsvChannelSpec("mavros-state.csv", "field.guided", "state.guided"),
    CsvChannelSpec("mavros-state.csv", "field.system_status", "state.system_status"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ALFA processed-data windows for UAVShield.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/alfa/extracted/processed/processed"),
        help="Directory containing extracted ALFA processed sequence folders.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/alfa/moment_windows/alfa_state_windows.npz"))
    parser.add_argument("--sample-period-s", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--boundary-margin-s",
        type=float,
        default=0.0,
        help="Drop windows whose label boundary is within this many seconds of either edge.",
    )
    parser.add_argument(
        "--min-post-failure-s",
        type=float,
        default=4.0,
        help=(
            "Minimum amount of post-failure evidence required to label a window anomalous. "
            "ALFA faults are sudden and often have short post-fault segments, so onset windows "
            "are valid anomaly examples when they contain enough post-fault data."
        ),
    )
    parser.add_argument("--include-no-ground-truth", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional max sequence count for smoke tests.")
    args = parser.parse_args()

    sequences = discover_sequences(args.root, include_no_ground_truth=args.include_no_ground_truth)
    if args.limit > 0:
        sequences = sequences[: args.limit]
    if not sequences:
        raise SystemExit(f"No ALFA sequence directories found under {args.root}")

    records: list[WindowRecord] = []
    failures: list[dict[str, str]] = []
    for sequence in sequences:
        try:
            records.extend(
                extract_sequence_windows(
                    sequence,
                    sample_period_s=args.sample_period_s,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    boundary_margin_s=args.boundary_margin_s,
                    min_post_failure_s=args.min_post_failure_s,
                )
            )
        except Exception as exc:
            failures.append({"sequence": sequence.name, "error": f"{type(exc).__name__}: {exc}"})

    if not records:
        raise SystemExit("No windows were extracted from ALFA. Check extraction and filters.")

    x = np.stack([record.x for record in records]).astype(np.float32)
    y = np.asarray([record.y for record in records], dtype=np.int64)
    labels = np.asarray([record.class_name for record in records])
    sequences_arr = np.asarray([record.sequence for record in records])
    paths = np.asarray([record.path for record in records])
    starts = np.asarray([record.window_start_s for record in records], dtype=np.float32)
    roles = np.asarray([record.role for record in records])
    failure_starts = np.asarray(
        [np.nan if record.failure_start_s is None else record.failure_start_s for record in records],
        dtype=np.float32,
    )
    failure_topics = np.asarray([record.failure_topics for record in records])

    normal_mask = y == 0
    if not np.any(normal_mask):
        raise SystemExit("At least one normal window is required for normalization.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        channel_mean = np.nanmean(x[normal_mask], axis=(0, 2), keepdims=True)
        channel_std = np.nanstd(x[normal_mask], axis=(0, 2), keepdims=True)
    channel_mean = np.nan_to_num(channel_mean, nan=0.0, posinf=0.0, neginf=0.0)
    channel_std = np.nan_to_num(channel_std, nan=1.0, posinf=1.0, neginf=1.0)
    channel_std[channel_std < 1e-6] = 1.0
    x_norm = np.nan_to_num((x - channel_mean) / channel_std, nan=0.0, posinf=0.0, neginf=0.0)
    x_raw = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=x_norm.astype(np.float32),
        x_raw=x_raw.astype(np.float32),
        y=y,
        classes=labels,
        labels=labels,
        file_keys=sequences_arr,
        files=paths,
        window_start_s=starts,
        roles=roles,
        failure_start_s=failure_starts,
        failure_topics=failure_topics,
        feature_names=np.asarray([channel.feature_name for channel in CHANNELS]),
        sample_period_s=np.asarray([args.sample_period_s], dtype=np.float32),
        seq_len=np.asarray([args.seq_len], dtype=np.int64),
        class_names=np.asarray(list(CLASS_TO_ID)),
        dataset=np.asarray(["ALFA"]),
    )
    write_manifest(args.output.with_suffix(".csv"), records)
    write_normalization(args.output.with_suffix(".normalization.json"), channel_mean, channel_std)
    write_summary(args.output.with_suffix(".summary.json"), sequences, records, failures, args)

    print(f"Wrote {len(records)} windows to {args.output}")
    print(f"Processed sequences: {len(sequences)}")
    for class_name in sorted(set(labels.tolist())):
        print(f"  {class_name}: {int((labels == class_name).sum())}")
    if failures:
        failure_path = args.output.parent / "alfa_window_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Failed {len(failures)} sequences; see {failure_path}")
    else:
        print("No parser failures")


def discover_sequences(root: Path, include_no_ground_truth: bool) -> list[SequenceInfo]:
    sequences: list[SequenceInfo] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if "no_ground_truth" in path.name and not include_no_ground_truth:
            continue

        failure_topics = tuple(sorted(failure_topics_for_sequence(path)))
        failure_start_s: float | None = None
        class_name = "Normal"
        if failure_topics:
            starts = [first_failure_time_s(path, topic) for topic in failure_topics]
            starts = [start for start in starts if start is not None]
            failure_start_s = min(starts) if starts else None
            class_names = sorted({FAILURE_TOPIC_TO_CLASS.get(topic, "Unknown Fault") for topic in failure_topics})
            if len(class_names) > 1:
                class_name = "Multiple Control Faults"
            else:
                class_name = class_names[0]
        elif "no_failure" not in path.name and "no_ground_truth" not in path.name:
            class_name = "Unknown Fault"

        sequences.append(
            SequenceInfo(
                path=path,
                name=path.name,
                class_name=class_name,
                failure_start_s=failure_start_s,
                failure_topics=failure_topics,
            )
        )
    return sequences


def extract_sequence_windows(
    sequence: SequenceInfo,
    sample_period_s: float,
    seq_len: int,
    stride: int,
    boundary_margin_s: float,
    min_post_failure_s: float,
) -> list[WindowRecord]:
    start_ns, end_ns = sequence_time_bounds(sequence.path)
    if end_ns <= start_ns:
        return []

    duration_s = (end_ns - start_ns) / 1e9
    grid_s = np.arange(0.0, duration_s, sample_period_s, dtype=np.float64)
    if grid_s.size < seq_len:
        return []

    series = np.vstack([interpolate_channel(sequence.path, channel, start_ns, grid_s) for channel in CHANNELS])
    records: list[WindowRecord] = []
    for start_idx in range(0, grid_s.size - seq_len + 1, stride):
        window_start_s = float(start_idx * sample_period_s)
        window_end_s = window_start_s + sample_period_s * (seq_len - 1)
        role, class_name = label_window(
            sequence,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            boundary_margin_s=boundary_margin_s,
            min_post_failure_s=min_post_failure_s,
        )
        if role == "boundary":
            continue
        y = CLASS_TO_ID.get(class_name, CLASS_TO_ID["Unknown Fault"])
        records.append(
            WindowRecord(
                x=series[:, start_idx : start_idx + seq_len],
                y=y,
                class_name=class_name,
                sequence=sequence.name,
                path=str(sequence.path),
                window_start_s=window_start_s,
                role=role,
                failure_start_s=sequence.failure_start_s,
                failure_topics=",".join(sequence.failure_topics),
            )
        )
    return records


def label_window(
    sequence: SequenceInfo,
    window_start_s: float,
    window_end_s: float,
    boundary_margin_s: float,
    min_post_failure_s: float,
) -> tuple[str, str]:
    if sequence.failure_start_s is None:
        return "normal", "Normal" if sequence.class_name != "Unknown Fault" else "Unknown Fault"

    failure_s = sequence.failure_start_s
    if window_end_s < failure_s - boundary_margin_s:
        return "pre_failure_normal", "Normal"
    if window_start_s > failure_s + boundary_margin_s:
        return "post_failure", sequence.class_name
    if window_start_s <= failure_s <= window_end_s and (window_end_s - failure_s) >= min_post_failure_s:
        return "fault_onset", sequence.class_name
    return "boundary", sequence.class_name


def sequence_time_bounds(sequence_path: Path) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for channel in CHANNELS:
        csv_path = csv_path_for(sequence_path, channel.suffix)
        if not csv_path.exists():
            continue
        timestamps, _ = read_csv_column(csv_path, channel.column)
        if timestamps.size:
            starts.append(float(np.nanmin(timestamps)))
            ends.append(float(np.nanmax(timestamps)))
    if not starts or not ends:
        return 0.0, 0.0
    return min(starts), max(ends)


def interpolate_channel(
    sequence_path: Path,
    channel: CsvChannelSpec,
    start_ns: float,
    grid_s: np.ndarray,
) -> np.ndarray:
    csv_path = csv_path_for(sequence_path, channel.suffix)
    if not csv_path.exists():
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    timestamps, values = read_csv_column(csv_path, channel.column)
    if timestamps.size < 2:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    mask = np.isfinite(timestamps) & np.isfinite(values)
    timestamps = timestamps[mask]
    values = values[mask]
    if timestamps.size < 2:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    order = np.argsort(timestamps)
    timestamps_s = (timestamps[order] - start_ns) / 1e9
    values = transform_values(values[order], channel.transform)

    unique_t, unique_indices = np.unique(timestamps_s, return_index=True)
    unique_values = values[unique_indices]
    if unique_t.size < 2:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)
    return np.interp(grid_s, unique_t, unique_values, left=np.nan, right=np.nan)


def read_csv_column(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "%time" not in (reader.fieldnames or []) or column not in (reader.fieldnames or []):
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        for row in reader:
            try:
                timestamp = float(row["%time"])
                value = parse_float(row[column])
            except (TypeError, ValueError):
                continue
            if math.isfinite(timestamp) and math.isfinite(value):
                timestamps.append(timestamp)
                values.append(value)
    return np.asarray(timestamps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def parse_float(value: str) -> float:
    value = value.strip()
    if value == "":
        return math.nan
    if value.lower() in {"true", "false"}:
        return 1.0 if value.lower() == "true" else 0.0
    return float(value)


def transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "percent":
        if values.size and np.nanmax(values) <= 1.5:
            return values * 100.0
        return values
    return values


def csv_path_for(sequence_path: Path, suffix: str) -> Path:
    return sequence_path / f"{sequence_path.name}-{suffix}"


def failure_topics_for_sequence(sequence_path: Path) -> list[str]:
    topics: list[str] = []
    prefix = f"{sequence_path.name}-failure_status-"
    for path in sequence_path.glob(f"{sequence_path.name}-failure_status-*.csv"):
        topic = path.name.removeprefix(prefix).removesuffix(".csv")
        topics.append(topic)
    return topics


def first_failure_time_s(sequence_path: Path, topic: str) -> float | None:
    csv_path = csv_path_for(sequence_path, f"failure_status-{topic}.csv")
    if not csv_path.exists():
        return None
    timestamps, values = read_csv_column(csv_path, "field.data")
    if timestamps.size == 0:
        return None
    positive = timestamps[values > 0]
    if positive.size == 0:
        return None
    start_ns, _ = sequence_time_bounds(sequence_path)
    if start_ns <= 0:
        return None
    return float((np.nanmin(positive) - start_ns) / 1e9)


def write_manifest(path: Path, records: list[WindowRecord]) -> None:
    fields = [
        "class_name",
        "y",
        "sequence",
        "path",
        "window_start_s",
        "role",
        "failure_start_s",
        "failure_topics",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "class_name": record.class_name,
                    "y": record.y,
                    "sequence": record.sequence,
                    "path": record.path,
                    "window_start_s": round(record.window_start_s, 3),
                    "role": record.role,
                    "failure_start_s": "" if record.failure_start_s is None else round(record.failure_start_s, 3),
                    "failure_topics": record.failure_topics,
                }
            )


def write_normalization(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    payload = {
        "feature_names": [channel.feature_name for channel in CHANNELS],
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
        "source": "ALFA processed normal and pre-failure-normal windows",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(
    path: Path,
    sequences: list[SequenceInfo],
    records: list[WindowRecord],
    failures: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    class_counts: dict[str, int] = {}
    sequence_counts: dict[str, int] = {}
    for sequence in sequences:
        sequence_counts[sequence.class_name] = sequence_counts.get(sequence.class_name, 0) + 1
    for record in records:
        class_counts[record.class_name] = class_counts.get(record.class_name, 0) + 1
    payload = {
        "dataset": "ALFA",
        "root": str(args.root),
        "sequences": len(sequences),
        "sequence_counts": sequence_counts,
        "windows": len(records),
        "window_counts": class_counts,
        "features": len(CHANNELS),
        "sample_period_s": args.sample_period_s,
        "seq_len": args.seq_len,
        "window_duration_s": args.sample_period_s * args.seq_len,
        "stride": args.stride,
        "boundary_margin_s": args.boundary_margin_s,
        "min_post_failure_s": args.min_post_failure_s,
        "failures": failures,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyulog import ULog


CLASS_TO_ID = {
    "Normal": 0,
    "External Position": 1,
    "Global Position": 2,
    "Altitude": 3,
    "Mechanical": 4,
    "Uncategorized": 5,
}

DEFAULT_EXCLUDED_FILE_KEYS = {
    # Present in mapping.json but absent from the upstream Hugging Face dataset.
    "2018-07-04/19_38_27",
    # Present upstream as zero-byte placeholder files, not parseable ULog data.
    "2019-01-22/07_55_19",
    "2019-03-04/08_04_13",
}


@dataclass(frozen=True)
class ChannelSpec:
    topic: str
    field: str
    feature_name: str
    transform: str = "identity"


CHANNELS = [
    ChannelSpec("vehicle_local_position", "x", "position.local_x_m"),
    ChannelSpec("vehicle_local_position", "y", "position.local_y_m"),
    ChannelSpec("vehicle_local_position", "z", "position.alt_m", "negate"),
    ChannelSpec("vehicle_local_position", "vx", "velocity.vx_mps"),
    ChannelSpec("vehicle_local_position", "vy", "velocity.vy_mps"),
    ChannelSpec("vehicle_local_position", "vz", "velocity.vz_mps"),
    ChannelSpec("vehicle_local_position", "eph", "position.eph_m"),
    ChannelSpec("vehicle_local_position", "epv", "position.epv_m"),
    ChannelSpec("vehicle_local_position", "heading", "position.heading_rad"),
    ChannelSpec("vehicle_local_position", "dist_bottom", "sensor.dist_bottom_m"),
    ChannelSpec("vehicle_global_position", "lat", "position.global_lat"),
    ChannelSpec("vehicle_global_position", "lon", "position.global_lon"),
    ChannelSpec("vehicle_global_position", "alt", "position.global_alt_m"),
    ChannelSpec("vehicle_global_position", "vel_n", "velocity.global_n_mps"),
    ChannelSpec("vehicle_global_position", "vel_e", "velocity.global_e_mps"),
    ChannelSpec("vehicle_global_position", "vel_d", "velocity.global_d_mps"),
    ChannelSpec("sensor_combined", "accelerometer_m_s2[0]", "acceleration.ax_mps2"),
    ChannelSpec("sensor_combined", "accelerometer_m_s2[1]", "acceleration.ay_mps2"),
    ChannelSpec("sensor_combined", "accelerometer_m_s2[2]", "acceleration.az_mps2"),
    ChannelSpec("sensor_combined", "gyro_rad[0]", "sensor.gyro_x_rad_s"),
    ChannelSpec("sensor_combined", "gyro_rad[1]", "sensor.gyro_y_rad_s"),
    ChannelSpec("sensor_combined", "gyro_rad[2]", "sensor.gyro_z_rad_s"),
    ChannelSpec("sensor_combined", "magnetometer_ga[0]", "sensor.mag_x_ga"),
    ChannelSpec("sensor_combined", "magnetometer_ga[1]", "sensor.mag_y_ga"),
    ChannelSpec("sensor_combined", "magnetometer_ga[2]", "sensor.mag_z_ga"),
    ChannelSpec("sensor_combined", "baro_alt_meter", "sensor.baro_alt_m"),
    ChannelSpec("battery_status", "remaining", "battery.percent", "percent"),
    ChannelSpec("battery_status", "current_a", "battery.current_a"),
    ChannelSpec("battery_status", "current_average_a", "battery.current_average_a"),
    ChannelSpec("battery_status", "voltage_v", "battery.voltage_v"),
    ChannelSpec("battery_status", "voltage_filtered_v", "battery.voltage_filtered_v"),
    ChannelSpec("battery_status", "discharged_mah", "battery.discharged_mah"),
    ChannelSpec("battery_status", "warning", "battery.warning"),
    ChannelSpec("distance_sensor", "current_distance", "ground_distance_m"),
    ChannelSpec("distance_sensor", "signal_quality", "sensor.distance_signal_quality"),
    ChannelSpec("vehicle_attitude", "rollspeed", "attitude.roll_rate_deg_s", "rad_to_deg"),
    ChannelSpec("vehicle_attitude", "pitchspeed", "attitude.pitch_rate_deg_s", "rad_to_deg"),
    ChannelSpec("vehicle_attitude", "yawspeed", "attitude.yaw_rate_deg_s", "rad_to_deg"),
    ChannelSpec("estimator_status", "pos_test_ratio", "estimator.pos_test_ratio"),
    ChannelSpec("estimator_status", "vel_test_ratio", "estimator.vel_test_ratio"),
    ChannelSpec("estimator_status", "hgt_test_ratio", "estimator.hgt_test_ratio"),
    ChannelSpec("estimator_status", "mag_test_ratio", "estimator.mag_test_ratio"),
    ChannelSpec("estimator_status", "hagl_test_ratio", "estimator.hagl_test_ratio"),
    ChannelSpec("estimator_status", "time_slip", "estimator.time_slip"),
    ChannelSpec("estimator_status", "innovation_check_flags", "estimator.innovation_check_flags"),
    ChannelSpec("estimator_status", "gps_check_fail_flags", "estimator.gps_check_fail_flags"),
    ChannelSpec("estimator_status", "timeout_flags", "estimator.timeout_flags"),
    ChannelSpec("estimator_status", "health_flags", "estimator.health_flags"),
    ChannelSpec("estimator_status", "vibe[0]", "estimator.vibe_x"),
    ChannelSpec("estimator_status", "vibe[1]", "estimator.vibe_y"),
    ChannelSpec("estimator_status", "vibe[2]", "estimator.vibe_z"),
    ChannelSpec("actuator_controls_0", "control[0]", "actuator.control_0"),
    ChannelSpec("actuator_controls_0", "control[1]", "actuator.control_1"),
    ChannelSpec("actuator_controls_0", "control[2]", "actuator.control_2"),
    ChannelSpec("actuator_controls_0", "control[3]", "actuator.control_3"),
    ChannelSpec("actuator_outputs", "output[0]", "actuator.output_0"),
    ChannelSpec("actuator_outputs", "output[1]", "actuator.output_1"),
    ChannelSpec("actuator_outputs", "output[2]", "actuator.output_2"),
    ChannelSpec("actuator_outputs", "output[3]", "actuator.output_3"),
    ChannelSpec("mission_result", "seq_reached", "mission.seq_reached"),
    ChannelSpec("mission_result", "finished", "mission.finished"),
    ChannelSpec("mission_result", "valid", "mission.valid"),
    ChannelSpec("position_setpoint_triplet", "current.x", "command.current_x_m"),
    ChannelSpec("position_setpoint_triplet", "current.y", "command.current_y_m"),
    ChannelSpec("position_setpoint_triplet", "current.z", "command.current_z_m"),
    ChannelSpec("position_setpoint_triplet", "current.lat", "command.current_lat"),
    ChannelSpec("position_setpoint_triplet", "current.lon", "command.current_lon"),
    ChannelSpec("position_setpoint_triplet", "current.alt", "command.current_alt_m"),
    ChannelSpec("telemetry_status", "rssi", "telemetry.rssi"),
    ChannelSpec("telemetry_status", "remote_rssi", "telemetry.remote_rssi"),
    ChannelSpec("telemetry_status", "rxerrors", "telemetry.rxerrors"),
    ChannelSpec("telemetry_status", "rx_message_lost_rate", "telemetry.rx_message_lost_rate"),
    ChannelSpec("telemetry_status", "rx_rate_avg", "telemetry.rx_rate_avg"),
    ChannelSpec("telemetry_status", "tx_rate_avg", "telemetry.tx_rate_avg"),
]


@dataclass(frozen=True)
class WindowRecord:
    x: np.ndarray
    y: int
    class_name: str
    file_key: str
    path: str
    window_start_s: float
    role: str
    signal: str = ""
    range_start_us: int | None = None
    range_end_us: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build broad UAV-SEAD state windows for MOMENT-based CrossGuard sanity checking."
    )
    parser.add_argument("--root", type=Path, default=Path("data/uav_sead"))
    parser.add_argument("--mapping", type=Path, default=Path("data/uav_sead/mapping.json"))
    parser.add_argument("--output", type=Path, default=Path("data/uav_sead/moment_windows/uav_sead_state_windows.npz"))
    parser.add_argument("--sample-period-s", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Optional max annotated logs to process.")
    parser.add_argument("--max-normal-windows-per-log", type=int, default=4)
    parser.add_argument("--max-anomaly-windows-per-log", type=int, default=12)
    parser.add_argument("--include-uncategorized", action="store_true")
    parser.add_argument(
        "--disable-default-exclusions",
        action="store_true",
        help="Attempt to parse every mapping entry, including known upstream invalid files.",
    )
    parser.add_argument(
        "--include-log-level-anomalies",
        action="store_true",
        help="Use whole-log sampled windows for anomalous logs that have no precise ranges.",
    )
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    records: list[WindowRecord] = []
    failures: list[dict[str, str]] = []
    filtered_uncategorized = 0
    excluded_known_invalid = 0
    excluded_file_keys = set() if args.disable_default_exclusions else DEFAULT_EXCLUDED_FILE_KEYS

    items = list(mapping.items())
    if args.limit > 0:
        items = items[: args.limit]

    for file_key, payload in items:
        if file_key in excluded_file_keys:
            excluded_known_invalid += 1
            continue

        annotations = payload.get("annotations", [])
        if should_skip_annotations(annotations, args.include_uncategorized):
            filtered_uncategorized += 1
            continue

        path = args.root / "ulg_files" / f"{file_key}.ulg"
        if not path.exists():
            failures.append({"file_key": file_key, "error": "missing ULog file"})
            continue

        try:
            records.extend(
                extract_windows(
                    path=path,
                    file_key=file_key,
                    annotations=annotations,
                    sample_period_s=args.sample_period_s,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_normal_windows=args.max_normal_windows_per_log,
                    max_anomaly_windows=args.max_anomaly_windows_per_log,
                    include_log_level_anomalies=args.include_log_level_anomalies,
                )
            )
        except Exception as exc:
            failures.append({"file_key": file_key, "error": f"{type(exc).__name__}: {exc}"})

    if not records:
        raise SystemExit("No windows were extracted. Check mapping, ULog files, and filters.")

    x = np.stack([record.x for record in records]).astype(np.float32)
    y = np.asarray([record.y for record in records], dtype=np.int64)
    classes = np.asarray([record.class_name for record in records])
    file_keys = np.asarray([record.file_key for record in records])
    paths = np.asarray([record.path for record in records])
    starts = np.asarray([record.window_start_s for record in records], dtype=np.float32)
    roles = np.asarray([record.role for record in records])
    signals = np.asarray([record.signal for record in records])

    normal_mask = y == 0
    if not np.any(normal_mask):
        raise SystemExit("At least one Normal window is required for normalization.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        channel_mean = np.nanmean(x[normal_mask], axis=(0, 2), keepdims=True)
        channel_std = np.nanstd(x[normal_mask], axis=(0, 2), keepdims=True)
    channel_mean = np.nan_to_num(channel_mean, nan=0.0, posinf=0.0, neginf=0.0)
    channel_std = np.nan_to_num(channel_std, nan=1.0, posinf=1.0, neginf=1.0)
    channel_std[channel_std < 1e-6] = 1.0
    x_norm = np.nan_to_num((x - channel_mean) / channel_std, nan=0.0, posinf=0.0, neginf=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=x_norm.astype(np.float32),
        x_raw=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        y=y,
        classes=classes,
        labels=classes,
        file_keys=file_keys,
        files=paths,
        window_start_s=starts,
        roles=roles,
        signals=signals,
        feature_names=np.asarray([channel.feature_name for channel in CHANNELS]),
        sample_period_s=np.asarray([args.sample_period_s], dtype=np.float32),
        seq_len=np.asarray([args.seq_len], dtype=np.int64),
        class_names=np.asarray(list(CLASS_TO_ID)),
    )
    write_manifest(args.output.with_suffix(".csv"), records)
    write_normalization(args.output.with_suffix(".normalization.json"), channel_mean, channel_std)
    if failures:
        (args.output.parent / "uav_sead_window_failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
    else:
        failure_path = args.output.parent / "uav_sead_window_failures.json"
        if failure_path.exists():
            failure_path.unlink()

    print(f"Wrote {len(records)} windows to {args.output}")
    if filtered_uncategorized:
        print(f"Filtered {filtered_uncategorized} Uncategorized entries")
    if excluded_known_invalid:
        print(f"Excluded {excluded_known_invalid} known invalid upstream entries")
    for class_name in sorted(set(classes.tolist())):
        print(f"  {class_name}: {int((classes == class_name).sum())}")
    if failures:
        print(f"Failed {len(failures)} logs; see uav_sead_window_failures.json")
    else:
        print("No parser failures")


def should_skip_annotations(annotations: list[dict], include_uncategorized: bool) -> bool:
    if include_uncategorized:
        return False
    return all(annotation.get("class") == "Uncategorized" for annotation in annotations)


def extract_windows(
    path: Path,
    file_key: str,
    annotations: list[dict],
    sample_period_s: float,
    seq_len: int,
    stride: int,
    max_normal_windows: int,
    max_anomaly_windows: int,
    include_log_level_anomalies: bool,
) -> list[WindowRecord]:
    ulog = ULog(str(path), None, disable_str_exceptions=True)
    datasets = {dataset.name: dataset.data for dataset in ulog.data_list}
    start_us, end_us = log_time_bounds(datasets)
    if end_us <= start_us:
        return []

    grid_s = np.arange(0.0, (end_us - start_us) / 1_000_000.0, sample_period_s, dtype=np.float64)
    if grid_s.size < seq_len:
        return []

    series = np.vstack([interpolate_channel(datasets, channel, start_us, grid_s) for channel in CHANNELS])
    records: list[WindowRecord] = []

    for annotation in annotations:
        class_name = str(annotation.get("class", "Uncategorized"))
        y = CLASS_TO_ID.get(class_name, CLASS_TO_ID["Uncategorized"])
        precise_ranges = flatten_ranges(annotation)
        if y == 0:
            for start_idx in sampled_starts(grid_s.size, seq_len, stride, max_normal_windows):
                records.append(
                    make_record(series, start_idx, seq_len, y, class_name, file_key, path, sample_period_s, "normal")
                )
            continue

        anomaly_count = 0
        for signal, range_start_us, range_end_us in precise_ranges:
            start_idx = centered_start_idx(
                range_start_us,
                range_end_us,
                log_start_us=start_us,
                sample_period_s=sample_period_s,
                seq_len=seq_len,
                grid_size=grid_s.size,
            )
            records.append(
                make_record(
                    series,
                    start_idx,
                    seq_len,
                    y,
                    class_name,
                    file_key,
                    path,
                    sample_period_s,
                    "anomaly_range",
                    signal=signal,
                    range_start_us=range_start_us,
                    range_end_us=range_end_us,
                )
            )
            anomaly_count += 1
            if max_anomaly_windows > 0 and anomaly_count >= max_anomaly_windows:
                break

        if not precise_ranges and include_log_level_anomalies:
            for start_idx in sampled_starts(grid_s.size, seq_len, stride, max_anomaly_windows):
                records.append(
                    make_record(
                        series,
                        start_idx,
                        seq_len,
                        y,
                        class_name,
                        file_key,
                        path,
                        sample_period_s,
                        "anomaly_log",
                    )
                )

    return records


def make_record(
    series: np.ndarray,
    start_idx: int,
    seq_len: int,
    y: int,
    class_name: str,
    file_key: str,
    path: Path,
    sample_period_s: float,
    role: str,
    signal: str = "",
    range_start_us: int | None = None,
    range_end_us: int | None = None,
) -> WindowRecord:
    window = series[:, start_idx : start_idx + seq_len]
    return WindowRecord(
        x=window,
        y=y,
        class_name=class_name,
        file_key=file_key,
        path=str(path),
        window_start_s=float(start_idx * sample_period_s),
        role=role,
        signal=signal,
        range_start_us=range_start_us,
        range_end_us=range_end_us,
    )


def flatten_ranges(annotation: dict) -> list[tuple[str, int, int]]:
    flattened: list[tuple[str, int, int]] = []
    for signal, ranges in annotation.get("ranges", []):
        for start_us, end_us in ranges:
            flattened.append((str(signal), int(start_us), int(end_us)))
    return flattened


def sampled_starts(grid_size: int, seq_len: int, stride: int, limit: int) -> list[int]:
    starts = list(range(0, grid_size - seq_len + 1, stride))
    if limit <= 0 or len(starts) <= limit:
        return starts
    indices = np.linspace(0, len(starts) - 1, limit).round().astype(int)
    return [starts[int(index)] for index in indices]


def centered_start_idx(
    range_start_us: int,
    range_end_us: int,
    log_start_us: float,
    sample_period_s: float,
    seq_len: int,
    grid_size: int,
) -> int:
    center_s = (((range_start_us + range_end_us) / 2.0) - log_start_us) / 1_000_000.0
    start_idx = int(round(center_s / sample_period_s)) - seq_len // 2
    return max(0, min(start_idx, grid_size - seq_len))


def log_time_bounds(datasets: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for channel in CHANNELS:
        data = datasets.get(channel.topic)
        if not data:
            continue
        timestamp = finite_array(data.get("timestamp"))
        if timestamp.size:
            starts.append(float(np.nanmin(timestamp)))
            ends.append(float(np.nanmax(timestamp)))
    if not starts or not ends:
        return 0.0, 0.0
    return min(starts), max(ends)


def interpolate_channel(
    datasets: dict[str, dict[str, np.ndarray]],
    channel: ChannelSpec,
    start_us: float,
    grid_s: np.ndarray,
) -> np.ndarray:
    data = datasets.get(channel.topic)
    if not data or channel.field not in data:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    timestamps = np.asarray(data.get("timestamp"), dtype=np.float64)
    values = np.asarray(data.get(channel.field), dtype=np.float64)
    n = min(timestamps.size, values.size)
    if n < 2:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    timestamps = timestamps[:n]
    values = values[:n]
    mask = np.isfinite(timestamps) & np.isfinite(values)
    timestamps = timestamps[mask]
    values = values[mask]
    if timestamps.size < 2:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    order = np.argsort(timestamps)
    timestamps_s = (timestamps[order] - start_us) / 1_000_000.0
    values = transform_values(values[order], channel.transform)

    unique_t, unique_indices = np.unique(timestamps_s, return_index=True)
    unique_values = values[unique_indices]
    if unique_t.size < 2:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)
    return np.interp(grid_s, unique_t, unique_values, left=np.nan, right=np.nan)


def transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "percent":
        if values.size and np.nanmax(values) <= 1.5:
            return values * 100.0
        return values
    if transform == "negate":
        return -values
    if transform == "rad_to_deg":
        return values * (180.0 / np.pi)
    return values


def finite_array(value) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    return arr[np.isfinite(arr)]


def write_manifest(path: Path, records: list[WindowRecord]) -> None:
    fields = [
        "class_name",
        "y",
        "file_key",
        "path",
        "window_start_s",
        "role",
        "signal",
        "range_start_us",
        "range_end_us",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "class_name": record.class_name,
                    "y": record.y,
                    "file_key": record.file_key,
                    "path": record.path,
                    "window_start_s": round(record.window_start_s, 3),
                    "role": record.role,
                    "signal": record.signal,
                    "range_start_us": record.range_start_us or "",
                    "range_end_us": record.range_end_us or "",
                }
            )


def write_normalization(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    payload = {
        "feature_names": [channel.feature_name for channel in CHANNELS],
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
        "source": "UAV-SEAD Normal windows only",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

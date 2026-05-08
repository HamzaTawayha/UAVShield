from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyulog import ULog


DATASET_FOLDERS = {
    "normal_good": ("logs", 0),
    "unsatisfactory": ("anomaly_logs", 1),
    "crash": ("crash_logs", 2),
}

CHANNELS = [
    ("vehicle_local_position", "x", "local_x_m"),
    ("vehicle_local_position", "y", "local_y_m"),
    ("vehicle_local_position", "z", "local_z_m"),
    ("vehicle_local_position", "vx", "local_vx_mps"),
    ("vehicle_local_position", "vy", "local_vy_mps"),
    ("vehicle_local_position", "vz", "local_vz_mps"),
    ("vehicle_acceleration", "xyz[0]", "accel_x_mps2"),
    ("vehicle_acceleration", "xyz[1]", "accel_y_mps2"),
    ("vehicle_acceleration", "xyz[2]", "accel_z_mps2"),
    ("battery_status", "remaining", "battery_remaining_pct"),
    ("battery_status", "current_a", "battery_current_a"),
    ("battery_status", "voltage_v", "battery_voltage_v"),
]


@dataclass(frozen=True)
class WindowRecord:
    x: np.ndarray
    y: int
    label: str
    file: str
    start_s: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PX4 ULog files into MOMENT-ready multivariate time-series windows."
    )
    parser.add_argument("--root", type=Path, default=Path("data/px4_flight_review"))
    parser.add_argument("--output", type=Path, default=Path("data/px4_flight_review/moment_windows/px4_moment_windows.npz"))
    parser.add_argument("--sample-period-s", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--limit-per-class", type=int, default=60)
    parser.add_argument("--max-windows-per-log", type=int, default=20)
    args = parser.parse_args()

    records: list[WindowRecord] = []
    failures: list[dict[str, str]] = []

    for label, (folder, numeric_label) in DATASET_FOLDERS.items():
        paths = sorted((args.root / folder).glob("*.ulg"))
        if args.limit_per_class > 0:
            paths = paths[: args.limit_per_class]

        for path in paths:
            try:
                records.extend(
                    extract_log_windows(
                        path=path,
                        label=label,
                        numeric_label=numeric_label,
                        sample_period_s=args.sample_period_s,
                        seq_len=args.seq_len,
                        stride=args.stride,
                        max_windows=args.max_windows_per_log,
                    )
                )
            except Exception as exc:
                failures.append({"file": path.name, "label": label, "error": f"{type(exc).__name__}: {exc}"})

    if not records:
        raise SystemExit("No windows were extracted. Check the input data folders.")

    x = np.stack([record.x for record in records]).astype(np.float32)
    y = np.asarray([record.y for record in records], dtype=np.int64)
    labels = np.asarray([record.label for record in records])
    files = np.asarray([record.file for record in records])
    starts = np.asarray([record.start_s for record in records], dtype=np.float32)

    normal_mask = y == 0
    channel_mean = np.nanmean(x[normal_mask], axis=(0, 2), keepdims=True)
    channel_std = np.nanstd(x[normal_mask], axis=(0, 2), keepdims=True)
    channel_std[channel_std < 1e-6] = 1.0
    x_norm = (x - channel_mean) / channel_std
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=x_norm,
        x_raw=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        y=y,
        labels=labels,
        files=files,
        window_start_s=starts,
        feature_names=np.asarray([name for _, _, name in CHANNELS]),
        sample_period_s=np.asarray([args.sample_period_s], dtype=np.float32),
        seq_len=np.asarray([args.seq_len], dtype=np.int64),
        stride=np.asarray([args.stride], dtype=np.int64),
    )

    write_manifest(args.output.with_suffix(".csv"), records)
    write_stats(args.output.with_suffix(".normalization.json"), channel_mean, channel_std)
    if failures:
        (args.output.parent / "window_extraction_failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )

    print(f"Wrote {len(records)} windows to {args.output}")
    for label in DATASET_FOLDERS:
        print(f"  {label}: {sum(1 for record in records if record.label == label)}")
    print(f"Shape: x={x_norm.shape} (windows, channels, timesteps)")
    if failures:
        print(f"Skipped {len(failures)} logs; see window_extraction_failures.json")


def extract_log_windows(
    path: Path,
    label: str,
    numeric_label: int,
    sample_period_s: float,
    seq_len: int,
    stride: int,
    max_windows: int,
) -> list[WindowRecord]:
    ulog = ULog(str(path), None, disable_str_exceptions=True)
    datasets = {dataset.name: dataset.data for dataset in ulog.data_list}

    start_us, end_us = log_time_bounds(datasets)
    if end_us <= start_us:
        return []

    grid_s = np.arange(0.0, (end_us - start_us) / 1_000_000.0, sample_period_s, dtype=np.float64)
    if grid_s.size < seq_len:
        return []

    series = np.vstack([interpolate_channel(datasets, topic, field, start_us, grid_s) for topic, field, _ in CHANNELS])
    windows: list[WindowRecord] = []
    for start_idx in range(0, grid_s.size - seq_len + 1, stride):
        window = series[:, start_idx : start_idx + seq_len]
        if not np.isfinite(window).any():
            continue
        windows.append(
            WindowRecord(
                x=window,
                y=numeric_label,
                label=label,
                file=path.name,
                start_s=float(grid_s[start_idx]),
            )
        )
        if max_windows > 0 and len(windows) >= max_windows:
            break
    return windows


def log_time_bounds(datasets: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for topic, _, _ in CHANNELS:
        data = datasets.get(topic)
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
    topic: str,
    field: str,
    start_us: float,
    grid_s: np.ndarray,
) -> np.ndarray:
    data = datasets.get(topic)
    if not data or field not in data:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    timestamps = finite_array(data.get("timestamp"))
    values = finite_array(data.get(field))
    n = min(timestamps.size, values.size)
    if n < 2:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    timestamps = timestamps[:n]
    values = values[:n]
    order = np.argsort(timestamps)
    timestamps_s = (timestamps[order] - start_us) / 1_000_000.0
    values = values[order]

    unique_t, unique_indices = np.unique(timestamps_s, return_index=True)
    unique_values = values[unique_indices]
    if unique_t.size < 2:
        return np.full(grid_s.shape, np.nan, dtype=np.float64)

    interpolated = np.interp(grid_s, unique_t, unique_values, left=np.nan, right=np.nan)
    if topic == "battery_status" and field == "remaining" and np.nanmax(interpolated) <= 1.5:
        interpolated = interpolated * 100.0
    return interpolated


def finite_array(value) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    return arr[np.isfinite(arr)]


def write_manifest(path: Path, records: list[WindowRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "y", "file", "window_start_s"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "label": record.label,
                    "y": record.y,
                    "file": record.file,
                    "window_start_s": round(record.start_s, 3),
                }
            )


def write_stats(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    data = {
        "feature_names": [name for _, _, name in CHANNELS],
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
        "source": "normal_good windows only",
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

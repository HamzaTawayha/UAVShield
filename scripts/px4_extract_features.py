from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np
from pyulog import ULog


DATASET_FOLDERS = {
    "normal_good": "logs",
    "unsatisfactory": "anomaly_logs",
    "crash": "crash_logs",
}


USEFUL_TOPICS = {
    "battery_status",
    "sensor_combined",
    "sensor_gps",
    "vehicle_acceleration",
    "vehicle_angular_velocity",
    "vehicle_attitude",
    "vehicle_global_position",
    "vehicle_local_position",
    "vehicle_status",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ML-ready summary features from PX4 ULog files.")
    parser.add_argument("--root", type=Path, default=Path("data/px4_flight_review"))
    parser.add_argument("--output", type=Path, default=Path("data/px4_flight_review/metadata/feature_preview.csv"))
    parser.add_argument("--limit-per-class", type=int, default=50)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for label, folder in DATASET_FOLDERS.items():
        paths = sorted((args.root / folder).glob("*.ulg"))
        if args.limit_per_class > 0:
            paths = paths[: args.limit_per_class]
        for path in paths:
            rows.append(extract_one(path, label))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} feature rows to {args.output}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["label"])] = counts.get(str(row["label"]), 0) + 1
    for label, count in counts.items():
        print(f"  {label}: {count}")


def fieldnames() -> list[str]:
    return [
        "label",
        "file",
        "parse_status",
        "size_mb",
        "duration_s",
        "topic_count",
        "useful_topics",
        "has_battery",
        "has_gps",
        "has_local_position",
        "has_attitude",
        "has_acceleration",
        "max_ground_speed_mps",
        "mean_ground_speed_mps",
        "max_vertical_speed_mps",
        "max_accel_mps2",
        "mean_accel_mps2",
        "local_xy_span_m",
        "battery_start_pct",
        "battery_end_pct",
        "battery_drop_pct",
        "battery_drop_pct_per_min",
        "mean_current_a",
        "mean_voltage_v",
    ]


def extract_one(path: Path, label: str) -> dict[str, object]:
    base: dict[str, object] = {
        "label": label,
        "file": path.name,
        "parse_status": "ok",
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
    }
    try:
        ulog = ULog(str(path), None, disable_str_exceptions=True)
        datasets = {dataset.name: dataset.data for dataset in ulog.data_list}
        topics = sorted(datasets)
        useful = sorted(set(topics).intersection(USEFUL_TOPICS))

        base.update(
            {
                "duration_s": round(duration_seconds(datasets), 3),
                "topic_count": len(topics),
                "useful_topics": ";".join(useful),
                "has_battery": int("battery_status" in datasets),
                "has_gps": int("sensor_gps" in datasets or "vehicle_global_position" in datasets),
                "has_local_position": int("vehicle_local_position" in datasets),
                "has_attitude": int("vehicle_attitude" in datasets),
                "has_acceleration": int("vehicle_acceleration" in datasets),
            }
        )
        base.update(local_position_features(datasets.get("vehicle_local_position", {})))
        base.update(acceleration_features(datasets.get("vehicle_acceleration", {})))
        base.update(battery_features(datasets.get("battery_status", {}), float(base["duration_s"] or 0.0)))
    except Exception as exc:
        base["parse_status"] = f"error:{type(exc).__name__}:{exc}"

    for name in fieldnames():
        base.setdefault(name, "")
    return base


def duration_seconds(datasets: dict[str, dict[str, np.ndarray]]) -> float:
    starts: list[float] = []
    ends: list[float] = []
    for data in datasets.values():
        timestamp = as_array(data.get("timestamp"))
        if timestamp.size:
            starts.append(float(np.nanmin(timestamp)))
            ends.append(float(np.nanmax(timestamp)))
    if not starts or not ends:
        return 0.0
    return max(0.0, (max(ends) - min(starts)) / 1_000_000.0)


def local_position_features(data: dict[str, np.ndarray]) -> dict[str, object]:
    vx = as_array(data.get("vx"))
    vy = as_array(data.get("vy"))
    vz = as_array(data.get("vz"))
    x = as_array(data.get("x"))
    y = as_array(data.get("y"))

    ground_speed = magnitude2(vx, vy)
    span = ""
    if x.size and y.size:
        span = round(math.hypot(float(np.nanmax(x) - np.nanmin(x)), float(np.nanmax(y) - np.nanmin(y))), 3)

    return {
        "max_ground_speed_mps": round(float(np.nanmax(ground_speed)), 3) if ground_speed.size else "",
        "mean_ground_speed_mps": round(float(np.nanmean(ground_speed)), 3) if ground_speed.size else "",
        "max_vertical_speed_mps": round(float(np.nanmax(np.abs(vz))), 3) if vz.size else "",
        "local_xy_span_m": span,
    }


def acceleration_features(data: dict[str, np.ndarray]) -> dict[str, object]:
    ax = as_array(data.get("xyz[0]"))
    ay = as_array(data.get("xyz[1]"))
    az = as_array(data.get("xyz[2]"))
    accel = magnitude3(ax, ay, az)
    return {
        "max_accel_mps2": round(float(np.nanmax(accel)), 3) if accel.size else "",
        "mean_accel_mps2": round(float(np.nanmean(accel)), 3) if accel.size else "",
    }


def battery_features(data: dict[str, np.ndarray], duration_s: float) -> dict[str, object]:
    remaining = as_array(data.get("remaining"))
    current = as_array(data.get("current_a"))
    voltage = as_array(data.get("voltage_v"))

    start = end = drop = drop_rate = ""
    if remaining.size:
        pct = remaining.astype(float)
        if np.nanmax(pct) <= 1.5:
            pct = pct * 100.0
        start = round(float(first_finite(pct)), 3)
        end = round(float(last_finite(pct)), 3)
        drop = round(float(start) - float(end), 3)
        if duration_s > 0:
            drop_rate = round(float(drop) / (duration_s / 60.0), 3)

    return {
        "battery_start_pct": start,
        "battery_end_pct": end,
        "battery_drop_pct": drop,
        "battery_drop_pct_per_min": drop_rate,
        "mean_current_a": round(float(np.nanmean(current)), 3) if current.size else "",
        "mean_voltage_v": round(float(np.nanmean(voltage)), 3) if voltage.size else "",
    }


def as_array(value) -> np.ndarray:
    if value is None:
        return np.array([])
    arr = np.asarray(value, dtype=float)
    return arr[np.isfinite(arr)]


def magnitude2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(a.size, b.size)
    if n == 0:
        return np.array([])
    return np.sqrt(a[:n] ** 2 + b[:n] ** 2)


def magnitude3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = min(a.size, b.size, c.size)
    if n == 0:
        return np.array([])
    return np.sqrt(a[:n] ** 2 + b[:n] ** 2 + c[:n] ** 2)


def first_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else float("nan")


def last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float("nan")


if __name__ == "__main__":
    main()

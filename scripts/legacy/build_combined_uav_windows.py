from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_TO_ID = {
    "Normal": 0,
    "UAV-SEAD External Position": 1,
    "UAV-SEAD Global Position": 2,
    "UAV-SEAD Altitude": 3,
    "UAV-SEAD Mechanical": 4,
    "ALFA Engine Failure": 5,
    "ALFA Aileron Fault": 6,
    "ALFA Elevator Fault": 7,
    "ALFA Rudder Fault": 8,
    "ALFA Multiple Control Faults": 9,
    "ALFA Unknown Fault": 10,
}

UAV_SEAD_LABEL_MAP = {
    "Normal": "Normal",
    "External Position": "UAV-SEAD External Position",
    "Global Position": "UAV-SEAD Global Position",
    "Altitude": "UAV-SEAD Altitude",
    "Mechanical": "UAV-SEAD Mechanical",
}

ALFA_LABEL_MAP = {
    "Normal": "Normal",
    "Engine Failure": "ALFA Engine Failure",
    "Aileron Fault": "ALFA Aileron Fault",
    "Elevator Fault": "ALFA Elevator Fault",
    "Rudder Fault": "ALFA Rudder Fault",
    "Multiple Control Faults": "ALFA Multiple Control Faults",
    "Unknown Fault": "ALFA Unknown Fault",
}


@dataclass(frozen=True)
class CommonFeature:
    name: str
    uav_sead: str | None
    alfa: str | None
    uav_sead_transform: str = "identity"
    alfa_transform: str = "identity"


COMMON_FEATURES = [
    CommonFeature("position.local_x_m", "position.local_x_m", "position.local_x_m"),
    CommonFeature("position.local_y_m", "position.local_y_m", "position.local_y_m"),
    CommonFeature("position.alt_m", "position.alt_m", "position.local_z_m"),
    CommonFeature("velocity.x_mps", "velocity.vx_mps", "velocity.local_x_mps"),
    CommonFeature("velocity.y_mps", "velocity.vy_mps", "velocity.local_y_mps"),
    CommonFeature("velocity.z_mps", "velocity.vz_mps", "velocity.local_z_mps"),
    CommonFeature("position.global_lat", "position.global_lat", "position.global_lat"),
    CommonFeature("position.global_lon", "position.global_lon", "position.global_lon"),
    CommonFeature("position.global_alt_m", "position.global_alt_m", "position.global_alt_m"),
    CommonFeature("position.heading_deg", "position.heading_rad", "position.heading_deg", "rad_to_deg"),
    CommonFeature("acceleration.ax_mps2", "acceleration.ax_mps2", "acceleration.ax_mps2"),
    CommonFeature("acceleration.ay_mps2", "acceleration.ay_mps2", "acceleration.ay_mps2"),
    CommonFeature("acceleration.az_mps2", "acceleration.az_mps2", "acceleration.az_mps2"),
    CommonFeature("sensor.gyro_x_rad_s", "sensor.gyro_x_rad_s", "sensor.gyro_x_rad_s"),
    CommonFeature("sensor.gyro_y_rad_s", "sensor.gyro_y_rad_s", "sensor.gyro_y_rad_s"),
    CommonFeature("sensor.gyro_z_rad_s", "sensor.gyro_z_rad_s", "sensor.gyro_z_rad_s"),
    CommonFeature("sensor.mag_x", "sensor.mag_x_ga", "sensor.mag_x"),
    CommonFeature("sensor.mag_y", "sensor.mag_y_ga", "sensor.mag_y"),
    CommonFeature("sensor.mag_z", "sensor.mag_z_ga", "sensor.mag_z"),
    CommonFeature("battery.percent", "battery.percent", "battery.percent"),
    CommonFeature("battery.current_a", "battery.current_a", "battery.current_a"),
    CommonFeature("battery.voltage_v", "battery.voltage_v", "battery.voltage_v"),
    CommonFeature("attitude.roll_rate_deg_s", "attitude.roll_rate_deg_s", "attitude.roll_rate_rad_s", "identity", "rad_to_deg"),
    CommonFeature("attitude.pitch_rate_deg_s", "attitude.pitch_rate_deg_s", "attitude.pitch_rate_rad_s", "identity", "rad_to_deg"),
    CommonFeature("attitude.yaw_rate_deg_s", "attitude.yaw_rate_deg_s", "attitude.yaw_rate_rad_s", "identity", "rad_to_deg"),
    CommonFeature("actuator.output_0", "actuator.output_0", "actuator.output_0"),
    CommonFeature("actuator.output_1", "actuator.output_1", "actuator.output_1"),
    CommonFeature("actuator.output_2", "actuator.output_2", "actuator.output_2"),
    CommonFeature("actuator.output_3", "actuator.output_3", "actuator.output_3"),
    CommonFeature("command.local_x_m", "command.current_x_m", "command.local_x_m"),
    CommonFeature("command.local_y_m", "command.current_y_m", "command.local_y_m"),
    CommonFeature("command.local_z_m", "command.current_z_m", "command.local_z_m"),
    CommonFeature("command.global_lat", "command.current_lat", "command.global_lat"),
    CommonFeature("command.global_lon", "command.current_lon", "command.global_lon"),
    CommonFeature("command.global_alt_m", "command.current_alt_m", "command.global_alt_m"),
    CommonFeature("telemetry.rssi", "telemetry.rssi", "telemetry.rc_rssi"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a common-schema combined UAV-SEAD + ALFA window dataset."
    )
    parser.add_argument(
        "--uav-sead",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_windows.npz"),
    )
    parser.add_argument(
        "--alfa",
        type=Path,
        default=Path("data/alfa/moment_windows/alfa_state_windows_16s.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/combined_uav/moment_windows/uav_sead_alfa_common_16s.npz"),
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--include-dataset-indicators",
        action="store_true",
        help="Append constant one-hot dataset indicator channels to the physical common schema.",
    )
    args = parser.parse_args()

    uav = load_dataset(
        path=args.uav_sead,
        dataset_name="UAV-SEAD",
        label_map=UAV_SEAD_LABEL_MAP,
        feature_source="uav_sead",
        target_seq_len=args.seq_len,
    )
    alfa = load_dataset(
        path=args.alfa,
        dataset_name="ALFA",
        label_map=ALFA_LABEL_MAP,
        feature_source="alfa",
        target_seq_len=args.seq_len,
    )

    datasets = [uav, alfa]
    x_raw = np.concatenate([dataset["x_raw"] for dataset in datasets], axis=0)
    labels = np.concatenate([dataset["labels"] for dataset in datasets], axis=0)
    y = np.asarray([CLASS_TO_ID[str(label)] for label in labels], dtype=np.int64)
    files = np.concatenate([dataset["files"] for dataset in datasets], axis=0)
    file_keys = np.concatenate([dataset["file_keys"] for dataset in datasets], axis=0)
    starts = np.concatenate([dataset["window_start_s"] for dataset in datasets], axis=0)
    roles = np.concatenate([dataset["roles"] for dataset in datasets], axis=0)
    source_datasets = np.concatenate([dataset["dataset"] for dataset in datasets], axis=0)

    feature_names = [feature.name for feature in COMMON_FEATURES]
    if args.include_dataset_indicators:
        indicator = np.zeros((x_raw.shape[0], 2, x_raw.shape[2]), dtype=np.float32)
        indicator[source_datasets == "UAV-SEAD", 0, :] = 1.0
        indicator[source_datasets == "ALFA", 1, :] = 1.0
        x_raw = np.concatenate([x_raw, indicator], axis=1)
        feature_names.extend(["dataset.is_uav_sead", "dataset.is_alfa"])

    x_norm, normalization = normalize_per_dataset(x_raw, y, source_datasets, feature_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=x_norm.astype(np.float32),
        x_raw=np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        y=y,
        classes=labels,
        labels=labels,
        file_keys=file_keys,
        files=files,
        window_start_s=starts.astype(np.float32),
        roles=roles,
        source_dataset=source_datasets,
        dataset_ids=np.asarray([0 if name == "UAV-SEAD" else 1 for name in source_datasets], dtype=np.int64),
        dataset_names=np.asarray(["UAV-SEAD", "ALFA"]),
        feature_names=np.asarray(feature_names),
        sample_period_s=np.asarray([0.5], dtype=np.float32),
        seq_len=np.asarray([args.seq_len], dtype=np.int64),
        class_names=np.asarray(list(CLASS_TO_ID)),
    )
    write_manifest(args.output.with_suffix(".csv"), y, labels, source_datasets, files, starts, roles)
    args.output.with_suffix(".normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    write_summary(args.output.with_suffix(".summary.json"), args, y, labels, source_datasets, feature_names)

    print(f"Wrote combined dataset to {args.output}")
    print(f"  windows={x_norm.shape[0]}, features={x_norm.shape[1]}, seq_len={x_norm.shape[2]}")
    for dataset_name in ["UAV-SEAD", "ALFA"]:
        mask = source_datasets == dataset_name
        print(f"  {dataset_name}: windows={int(mask.sum())}, anomalies={int((y[mask] != 0).sum())}")
    for label in sorted(set(labels.tolist())):
        print(f"  {label}: {int((labels == label).sum())}")


def load_dataset(
    path: Path,
    dataset_name: str,
    label_map: dict[str, str],
    feature_source: str,
    target_seq_len: int,
) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    x_raw_source = data["x_raw"].astype(np.float32) if "x_raw" in data.files else data["x"].astype(np.float32)
    source_feature_names = [str(name) for name in data["feature_names"]]
    source_index = {name: idx for idx, name in enumerate(source_feature_names)}
    source_seq_len = x_raw_source.shape[2]
    if source_seq_len < target_seq_len:
        raise ValueError(f"{path} has seq_len={source_seq_len}, smaller than target seq_len={target_seq_len}")
    crop_start = (source_seq_len - target_seq_len) // 2
    crop_end = crop_start + target_seq_len

    mapped = np.full(
        (x_raw_source.shape[0], len(COMMON_FEATURES), target_seq_len),
        np.nan,
        dtype=np.float32,
    )
    for out_idx, feature in enumerate(COMMON_FEATURES):
        source_name = feature.uav_sead if feature_source == "uav_sead" else feature.alfa
        transform = feature.uav_sead_transform if feature_source == "uav_sead" else feature.alfa_transform
        if source_name is None or source_name not in source_index:
            continue
        values = x_raw_source[:, source_index[source_name], crop_start:crop_end]
        mapped[:, out_idx, :] = transform_values(values, transform)

    raw_labels = np.asarray([str(label) for label in data["labels"]])
    labels = np.asarray([label_map.get(label, f"{dataset_name} {label}") for label in raw_labels])
    files = np.asarray([f"{dataset_name}:{path_value}" for path_value in data["files"].astype(str)])
    file_keys = (
        np.asarray([f"{dataset_name}:{path_value}" for path_value in data["file_keys"].astype(str)])
        if "file_keys" in data.files
        else files
    )
    starts = data["window_start_s"].astype(np.float32) + np.float32(crop_start * 0.5)
    roles = (
        np.asarray([f"{dataset_name}:{role}" for role in data["roles"].astype(str)])
        if "roles" in data.files
        else np.asarray([dataset_name] * len(labels))
    )
    return {
        "x_raw": mapped,
        "labels": labels,
        "files": files,
        "file_keys": file_keys,
        "window_start_s": starts,
        "roles": roles,
        "dataset": np.asarray([dataset_name] * len(labels)),
    }


def transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "rad_to_deg":
        return values * np.float32(180.0 / np.pi)
    return values


def normalize_per_dataset(
    x_raw: np.ndarray,
    y: np.ndarray,
    source_datasets: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict]:
    x_norm = np.empty_like(x_raw, dtype=np.float32)
    payload = {
        "source": "Per-dataset normal-window normalization after common-schema mapping",
        "feature_names": feature_names,
        "datasets": {},
    }
    for dataset_name in sorted(set(source_datasets.tolist())):
        mask = source_datasets == dataset_name
        normal_mask = mask & (y == 0)
        if not np.any(normal_mask):
            raise ValueError(f"No normal windows available for {dataset_name} normalization.")
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(x_raw[normal_mask], axis=(0, 2), keepdims=True)
            std = np.nanstd(x_raw[normal_mask], axis=(0, 2), keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std[std < 1e-6] = 1.0
        x_norm[mask] = np.nan_to_num((x_raw[mask] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        payload["datasets"][dataset_name] = {
            "normal_windows": int(normal_mask.sum()),
            "mean": mean.reshape(-1).astype(float).tolist(),
            "std": std.reshape(-1).astype(float).tolist(),
        }
    return x_norm, payload


def write_manifest(
    path: Path,
    y: np.ndarray,
    labels: np.ndarray,
    source_datasets: np.ndarray,
    files: np.ndarray,
    starts: np.ndarray,
    roles: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "y", "source_dataset", "file", "window_start_s", "role"],
        )
        writer.writeheader()
        for idx in range(len(labels)):
            writer.writerow(
                {
                    "class_name": str(labels[idx]),
                    "y": int(y[idx]),
                    "source_dataset": str(source_datasets[idx]),
                    "file": str(files[idx]),
                    "window_start_s": round(float(starts[idx]), 3),
                    "role": str(roles[idx]),
                }
            )


def write_summary(
    path: Path,
    args: argparse.Namespace,
    y: np.ndarray,
    labels: np.ndarray,
    source_datasets: np.ndarray,
    feature_names: list[str],
) -> None:
    payload = {
        "sources": {
            "uav_sead": str(args.uav_sead),
            "alfa": str(args.alfa),
        },
        "windows": int(len(y)),
        "features": len(feature_names),
        "seq_len": args.seq_len,
        "sample_period_s": 0.5,
        "window_duration_s": args.seq_len * 0.5,
        "include_dataset_indicators": bool(args.include_dataset_indicators),
        "dataset_counts": {},
        "label_counts": {},
    }
    for dataset_name in sorted(set(source_datasets.tolist())):
        mask = source_datasets == dataset_name
        payload["dataset_counts"][dataset_name] = {
            "windows": int(mask.sum()),
            "normal_windows": int((y[mask] == 0).sum()),
            "anomaly_windows": int((y[mask] != 0).sum()),
        }
    for label in sorted(set(labels.tolist())):
        payload["label_counts"][label] = int((labels == label).sum())
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

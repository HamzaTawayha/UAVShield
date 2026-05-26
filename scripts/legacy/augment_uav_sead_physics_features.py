from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append UAV physics/consistency features to an existing UAV-SEAD window artifact."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_windows.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_physics_windows.npz"),
    )
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "x_raw" not in data.files:
        raise SystemExit(f"{args.input} does not contain x_raw; cannot build physical residual features.")

    x = data["x"].astype(np.float32)
    x_raw = data["x_raw"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = np.asarray([str(item) for item in data["feature_names"]])
    feature_index = {name: index for index, name in enumerate(feature_names.tolist())}

    derived_raw, derived_names = build_derived_features(x_raw, feature_index)
    normal_mask = y == 0
    if not np.any(normal_mask):
        raise SystemExit("At least one normal window is required to normalize derived features.")

    mean = np.nanmean(derived_raw[normal_mask], axis=(0, 2), keepdims=True)
    std = np.nanstd(derived_raw[normal_mask], axis=(0, 2), keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std < 1e-6] = 1.0
    derived_norm = np.nan_to_num((derived_raw - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    x_aug = np.concatenate([x, derived_norm.astype(np.float32)], axis=1)
    x_raw_aug = np.concatenate([x_raw, np.nan_to_num(derived_raw, nan=0.0, posinf=0.0, neginf=0.0)], axis=1)
    feature_names_aug = np.concatenate([feature_names, np.asarray(derived_names)])

    payload = {key: data[key] for key in data.files if key not in {"x", "x_raw", "feature_names"}}
    payload["x"] = x_aug.astype(np.float32)
    payload["x_raw"] = x_raw_aug.astype(np.float32)
    payload["feature_names"] = feature_names_aug

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    write_summary(args.output.with_suffix(".summary.json"), args.input, x.shape, x_aug.shape, derived_names)
    write_normalization(args.output.with_suffix(".derived_normalization.json"), derived_names, mean, std)

    print(f"Wrote augmented windows to {args.output}")
    print(f"  original_shape={x.shape}")
    print(f"  augmented_shape={x_aug.shape}")
    print(f"  added_features={len(derived_names)}")
    for name in derived_names:
        print(f"    {name}")


def build_derived_features(
    x_raw: np.ndarray,
    feature_index: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    builders: list[tuple[str, Callable[[], np.ndarray]]] = [
        ("derived.alt_local_minus_baro_m", lambda: channel(x_raw, feature_index, "position.alt_m") - channel(x_raw, feature_index, "sensor.baro_alt_m")),
        ("derived.alt_local_minus_distance_m", lambda: channel(x_raw, feature_index, "position.alt_m") - channel(x_raw, feature_index, "ground_distance_m")),
        ("derived.command_alt_minus_local_alt_m", lambda: channel(x_raw, feature_index, "command.current_alt_m") - channel(x_raw, feature_index, "position.alt_m")),
        ("derived.local_speed_norm_mps", lambda: norm_channels(x_raw, feature_index, ["velocity.vx_mps", "velocity.vy_mps", "velocity.vz_mps"])),
        ("derived.global_speed_norm_mps", lambda: norm_channels(x_raw, feature_index, ["velocity.global_n_mps", "velocity.global_e_mps", "velocity.global_d_mps"])),
        ("derived.speed_norm_delta_mps", lambda: norm_channels(x_raw, feature_index, ["velocity.vx_mps", "velocity.vy_mps", "velocity.vz_mps"]) - norm_channels(x_raw, feature_index, ["velocity.global_n_mps", "velocity.global_e_mps", "velocity.global_d_mps"])),
        ("derived.accel_norm_mps2", lambda: norm_channels(x_raw, feature_index, ["acceleration.ax_mps2", "acceleration.ay_mps2", "acceleration.az_mps2"])),
        ("derived.gyro_norm_rad_s", lambda: norm_channels(x_raw, feature_index, ["sensor.gyro_x_rad_s", "sensor.gyro_y_rad_s", "sensor.gyro_z_rad_s"])),
        ("derived.mag_norm_ga", lambda: norm_channels(x_raw, feature_index, ["sensor.mag_x_ga", "sensor.mag_y_ga", "sensor.mag_z_ga"])),
        ("derived.vibe_norm", lambda: norm_channels(x_raw, feature_index, ["estimator.vibe_x", "estimator.vibe_y", "estimator.vibe_z"])),
        ("derived.estimator_max_test_ratio", lambda: max_channels(x_raw, feature_index, ["estimator.pos_test_ratio", "estimator.vel_test_ratio", "estimator.hgt_test_ratio", "estimator.mag_test_ratio", "estimator.hagl_test_ratio"])),
        ("derived.estimator_flag_sum", lambda: sum_channels(x_raw, feature_index, ["estimator.innovation_check_flags", "estimator.gps_check_fail_flags", "estimator.timeout_flags", "estimator.health_flags"])),
        ("derived.actuator_control_norm", lambda: norm_channels(x_raw, feature_index, ["actuator.control_0", "actuator.control_1", "actuator.control_2", "actuator.control_3"])),
        ("derived.actuator_output_mean", lambda: mean_channels(x_raw, feature_index, ["actuator.output_0", "actuator.output_1", "actuator.output_2", "actuator.output_3"])),
        ("derived.actuator_output_spread", lambda: spread_channels(x_raw, feature_index, ["actuator.output_0", "actuator.output_1", "actuator.output_2", "actuator.output_3"])),
        ("derived.battery_power_w", lambda: channel(x_raw, feature_index, "battery.voltage_v") * channel(x_raw, feature_index, "battery.current_a")),
        ("derived.battery_voltage_delta_v", lambda: channel(x_raw, feature_index, "battery.voltage_v") - channel(x_raw, feature_index, "battery.voltage_filtered_v")),
        ("derived.telemetry_rssi_delta", lambda: channel(x_raw, feature_index, "telemetry.rssi") - channel(x_raw, feature_index, "telemetry.remote_rssi")),
    ]

    names: list[str] = []
    features: list[np.ndarray] = []
    for name, builder in builders:
        try:
            value = builder()
        except KeyError:
            continue
        names.append(name)
        features.append(value.astype(np.float32))

    if not features:
        raise SystemExit("No derived features could be built from the input feature names.")
    return np.stack(features, axis=1), names


def channel(x_raw: np.ndarray, feature_index: dict[str, int], name: str) -> np.ndarray:
    return x_raw[:, feature_index[name], :]


def stack_channels(x_raw: np.ndarray, feature_index: dict[str, int], names: list[str]) -> np.ndarray:
    return np.stack([channel(x_raw, feature_index, name) for name in names], axis=1)


def norm_channels(x_raw: np.ndarray, feature_index: dict[str, int], names: list[str]) -> np.ndarray:
    values = stack_channels(x_raw, feature_index, names)
    return np.sqrt(np.sum(values * values, axis=1))


def max_channels(x_raw: np.ndarray, feature_index: dict[str, int], names: list[str]) -> np.ndarray:
    return np.max(stack_channels(x_raw, feature_index, names), axis=1)


def mean_channels(x_raw: np.ndarray, feature_index: dict[str, int], names: list[str]) -> np.ndarray:
    return np.mean(stack_channels(x_raw, feature_index, names), axis=1)


def sum_channels(x_raw: np.ndarray, feature_index: dict[str, int], names: list[str]) -> np.ndarray:
    return np.sum(stack_channels(x_raw, feature_index, names), axis=1)


def spread_channels(x_raw: np.ndarray, feature_index: dict[str, int], names: list[str]) -> np.ndarray:
    values = stack_channels(x_raw, feature_index, names)
    return np.max(values, axis=1) - np.min(values, axis=1)


def write_summary(
    path: Path,
    input_path: Path,
    original_shape: tuple[int, ...],
    augmented_shape: tuple[int, ...],
    derived_names: list[str],
) -> None:
    summary = {
        "input": str(input_path),
        "original_shape": list(original_shape),
        "augmented_shape": list(augmented_shape),
        "added_feature_count": len(derived_names),
        "added_features": derived_names,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_normalization(path: Path, names: list[str], mean: np.ndarray, std: np.ndarray) -> None:
    payload = {
        "feature_names": names,
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_statistical_feature_baseline import (
    build_model,
    grouped_train_cal_test_split,
    print_split_summary,
    split_summary,
    statistical_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multiclass statistical-feature model and calibrate one decision "
            "threshold per UAV-SEAD anomaly family."
        )
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_physics_windows.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows/statistical_features_family_thresholds"),
    )
    parser.add_argument(
        "--model",
        choices=["hist_gb", "extra_trees", "random_forest"],
        default="hist_gb",
    )
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "accuracy"],
        default="f1",
        help="Metric optimized independently for each family on the calibration split.",
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if not 0.0 < args.threshold_step <= 1.0:
        raise SystemExit("--threshold-step must be in (0, 1].")

    data = np.load(args.windows, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    labels = np.asarray([str(item) for item in data["labels"]])
    files = data["files"]
    starts = data["window_start_s"]
    class_names = np.asarray([str(item) for item in data["class_names"]])
    feature_names = np.asarray([str(item) for item in data["feature_names"]])
    y_binary = (y != 0).astype(np.int64)

    x_features, stat_feature_names = statistical_features(x, feature_names)
    train_idx, cal_idx, test_idx = grouped_train_cal_test_split(
        y=y_binary,
        files=files,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
    )

    print(
        f"Dataset windows={len(y)}, stat_features={x_features.shape[1]}, "
        f"normal={int((y_binary == 0).sum())}, anomaly={int((y_binary == 1).sum())}"
    )
    print_split_summary("train", y_binary[train_idx])
    print_split_summary("calibration", y_binary[cal_idx])
    print_split_summary("test", y_binary[test_idx])

    model = build_model(args.model, args.seed)
    model.fit(x_features[train_idx], y[train_idx])

    cal_prob = aligned_probabilities(model, x_features[cal_idx], class_names.size)
    test_prob = aligned_probabilities(model, x_features[test_idx], class_names.size)
    anomaly_class_ids = [int(class_id) for class_id in sorted(np.unique(y)) if int(class_id) != 0]
    thresholds = calibrate_thresholds(
        probabilities=cal_prob,
        y_true=y[cal_idx],
        class_ids=anomaly_class_ids,
        class_names=class_names,
        step=args.threshold_step,
        metric=args.threshold_metric,
    )

    y_pred = predict_with_family_thresholds(test_prob, thresholds, anomaly_class_ids)
    evaluation = evaluate(
        y_true=y[test_idx],
        y_binary_true=y_binary[test_idx],
        y_pred=y_pred,
        probabilities=test_prob,
        labels=labels[test_idx],
        files=files[test_idx],
        starts=starts[test_idx],
        class_names=class_names,
        thresholds=thresholds,
        anomaly_class_ids=anomaly_class_ids,
    )
    evaluation["model"] = f"Statistical window features + {args.model} + per-family thresholds"
    evaluation["threshold_strategy"] = f"per_family_{args.threshold_metric}_grid"
    evaluation["threshold"] = {class_names[class_id]: thresholds[class_id]["threshold"] for class_id in anomaly_class_ids}
    evaluation["interpretation"] = (
        "P(family) >= calibrated family threshold triggers that family; no crossing means Normal."
    )
    evaluation["train"] = split_summary(y_binary[train_idx])
    evaluation["calibration"] = split_summary(y_binary[cal_idx])
    evaluation["test"]["windows"] = int(len(test_idx))
    evaluation["hyperparameters"] = {
        "windows": str(args.windows),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "seed": args.seed,
        "threshold_step": args.threshold_step,
        "threshold_metric": args.threshold_metric,
        "split": {
            "test_size": args.test_size,
            "calibration_size": args.calibration_size,
            "test_random_state": 11,
            "calibration_random_state": 13,
            "grouped_by": "files",
        },
        "feature_builder": {
            "stats": ["mean", "std", "min", "max", "last_minus_first", "q10", "q90", "energy"],
            "source_channels": int(x.shape[1]),
            "stat_features": int(x_features.shape[1]),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "thresholds": thresholds,
            "class_names": class_names.tolist(),
            "stat_feature_names": stat_feature_names,
            "source_feature_names": feature_names.tolist(),
        },
        args.output_dir / "family_threshold_model.joblib",
    )
    (args.output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(evaluation["hyperparameters"], indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "family_thresholds.json").write_text(
        json.dumps({class_names[class_id]: thresholds[class_id] for class_id in anomaly_class_ids}, indent=2),
        encoding="utf-8",
    )
    write_thresholds_csv(args.output_dir / "family_thresholds.csv", thresholds, class_names, anomaly_class_ids)
    write_predictions(args.output_dir / "predictions.csv", evaluation["predictions"], class_names)

    print(f"\nWrote model to {args.output_dir / 'family_threshold_model.joblib'}")
    print_result(evaluation, class_names, anomaly_class_ids)


def aligned_probabilities(model: Any, x: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(x)
    probabilities = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    for column, class_id in enumerate(model.classes_):
        probabilities[:, int(class_id)] = raw[:, column]
    return probabilities


def calibrate_thresholds(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    class_ids: list[int],
    class_names: np.ndarray,
    step: float,
    metric: str,
) -> dict[int, dict[str, Any]]:
    grid = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    thresholds: dict[int, dict[str, Any]] = {}
    for class_id in class_ids:
        target = (y_true == class_id).astype(np.int64)
        scores = probabilities[:, class_id]
        best_key = (-1.0, -1.0, -1.0)
        best_payload: dict[str, Any] | None = None
        for threshold in grid:
            pred = (scores >= threshold).astype(np.int64)
            precision, recall, f1, _ = precision_recall_fscore_support(
                target,
                pred,
                average="binary",
                zero_division=0,
            )
            accuracy = accuracy_score(target, pred)
            key = (float(f1), float(accuracy), float(recall)) if metric == "f1" else (float(accuracy), float(f1), float(recall))
            if key > best_key:
                best_key = key
                best_payload = {
                    "class_id": int(class_id),
                    "class_name": str(class_names[class_id]),
                    "threshold": float(threshold),
                    "calibration_accuracy": float(accuracy),
                    "calibration_precision": float(precision),
                    "calibration_recall": float(recall),
                    "calibration_f1": float(f1),
                    "calibration_positive_windows": int(target.sum()),
                }
        if best_payload is None:
            raise RuntimeError(f"No threshold selected for {class_names[class_id]}")
        thresholds[class_id] = best_payload
    return thresholds


def predict_with_family_thresholds(
    probabilities: np.ndarray,
    thresholds: dict[int, dict[str, Any]],
    class_ids: list[int],
) -> np.ndarray:
    predictions = np.zeros(probabilities.shape[0], dtype=np.int64)
    for row_index, row in enumerate(probabilities):
        candidates: list[tuple[float, float, int]] = []
        for class_id in class_ids:
            threshold = float(thresholds[class_id]["threshold"])
            probability = float(row[class_id])
            if probability >= threshold:
                candidates.append((probability - threshold, probability, class_id))
        if candidates:
            predictions[row_index] = max(candidates)[2]
    return predictions


def evaluate(
    y_true: np.ndarray,
    y_binary_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    files: np.ndarray,
    starts: np.ndarray,
    class_names: np.ndarray,
    thresholds: dict[int, dict[str, Any]],
    anomaly_class_ids: list[int],
) -> dict[str, Any]:
    y_binary_pred = (y_pred != 0).astype(np.int64)
    anomaly_score = 1.0 - probabilities[:, 0]
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        y_binary_true,
        y_binary_pred,
        average="binary",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    try:
        roc_auc = roc_auc_score(y_binary_true, anomaly_score)
    except ValueError:
        roc_auc = 0.0

    normal_mask = y_binary_true == 0
    anomaly_mask = y_binary_true == 1
    by_label: dict[str, dict[str, float]] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        label_pred = y_binary_pred[mask]
        by_label[str(label)] = {
            "n": int(mask.sum()),
            "mean_score": float(anomaly_score[mask].mean()) if np.any(mask) else 0.0,
            "flagged_count": int(label_pred.sum()),
            "flagged_rate": float(label_pred.mean()) if label_pred.size else 0.0,
        }

    per_family: dict[str, dict[str, float]] = {}
    for class_id in anomaly_class_ids:
        class_mask = y_true == class_id
        predicted_mask = y_pred == class_id
        non_family_mask = y_true != class_id
        normal_predicted_mask = normal_mask & predicted_mask
        precision, recall, f1, _ = precision_recall_fscore_support(
            (y_true == class_id).astype(np.int64),
            predicted_mask.astype(np.int64),
            average="binary",
            zero_division=0,
        )
        per_family[str(class_names[class_id])] = {
            "threshold": float(thresholds[class_id]["threshold"]),
            "test_true_windows": int(class_mask.sum()),
            "test_predicted_windows": int(predicted_mask.sum()),
            "test_detected_windows": int(np.sum(class_mask & predicted_mask)),
            "test_detection_rate": float(np.mean(predicted_mask[class_mask])) if np.any(class_mask) else 0.0,
            "test_false_positive_rate_vs_non_family": float(np.mean(predicted_mask[non_family_mask])) if np.any(non_family_mask) else 0.0,
            "test_normal_false_family_rate": float(np.mean(normal_predicted_mask[normal_mask])) if np.any(normal_mask) else 0.0,
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "calibration_accuracy": float(thresholds[class_id]["calibration_accuracy"]),
            "calibration_precision": float(thresholds[class_id]["calibration_precision"]),
            "calibration_recall": float(thresholds[class_id]["calibration_recall"]),
            "calibration_f1": float(thresholds[class_id]["calibration_f1"]),
        }

    predictions = []
    for label, true, pred, file, start, row, score, flagged in zip(
        labels,
        y_true,
        y_pred,
        files,
        starts,
        probabilities,
        anomaly_score,
        y_binary_pred,
    ):
        predictions.append(
            {
                "label": str(label),
                "y_true_class": int(true),
                "y_true_binary": int(true != 0),
                "predicted_class": int(pred),
                "predicted_label": str(class_names[pred]),
                "file": str(file),
                "window_start_s": float(start),
                "anomaly_score": float(score),
                "flagged": int(flagged),
                **{f"prob_{safe_name(class_names[class_id])}": float(row[class_id]) for class_id in range(len(class_names))},
            }
        )

    return {
        "test": {
            "accuracy": float(accuracy_score(y_binary_true, y_binary_pred)),
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "f1": float(binary_f1),
            "roc_auc": float(roc_auc),
            "normal_flag_rate": float(np.mean(y_binary_pred[normal_mask])) if np.any(normal_mask) else 0.0,
            "anomaly_flag_rate": float(np.mean(y_binary_pred[anomaly_mask])) if np.any(anomaly_mask) else 0.0,
            "multiclass_accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
        },
        "by_label": by_label,
        "per_family": per_family,
        "predictions": predictions,
    }


def write_thresholds_csv(
    path: Path,
    thresholds: dict[int, dict[str, Any]],
    class_names: np.ndarray,
    anomaly_class_ids: list[int],
) -> None:
    fields = [
        "class_id",
        "class_name",
        "threshold",
        "calibration_accuracy",
        "calibration_precision",
        "calibration_recall",
        "calibration_f1",
        "calibration_positive_windows",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for class_id in anomaly_class_ids:
            writer.writerow({field: thresholds[class_id].get(field, "") for field in fields})


def write_predictions(path: Path, predictions: list[dict[str, Any]], class_names: np.ndarray) -> None:
    fields = [
        "label",
        "y_true_class",
        "y_true_binary",
        "predicted_class",
        "predicted_label",
        "file",
        "window_start_s",
        "anomaly_score",
        "flagged",
        *[f"prob_{safe_name(name)}" for name in class_names],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def print_result(evaluation: dict[str, Any], class_names: np.ndarray, anomaly_class_ids: list[int]) -> None:
    test = evaluation["test"]
    print("Per-family thresholds:")
    for class_id in anomaly_class_ids:
        stats = evaluation["per_family"][str(class_names[class_id])]
        print(
            f"  {class_names[class_id]}: threshold={stats['threshold']:.2f}, "
            f"test_detection={stats['test_detection_rate']:.2%}, "
            f"test_precision={stats['test_precision']:.4f}, test_f1={stats['test_f1']:.4f}"
        )
    print("Overall binary defense metrics:")
    print(f"Test ROC-AUC: {test['roc_auc']:.4f}")
    print(f"Test accuracy: {test['accuracy']:.4f}")
    print(f"Test precision: {test['precision']:.4f}")
    print(f"Test recall: {test['recall']:.4f}")
    print(f"Test F1: {test['f1']:.4f}")
    print(f"Test normal flag rate: {test['normal_flag_rate']:.2%}")
    print(f"Test anomaly flag rate: {test['anomaly_flag_rate']:.2%}")
    print(f"Test multiclass accuracy: {test['multiclass_accuracy']:.4f}")
    print(f"Test macro F1: {test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

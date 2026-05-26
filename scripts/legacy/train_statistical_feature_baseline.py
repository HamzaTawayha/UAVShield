from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tabular telemetry-statistics anomaly baseline on UAV window artifacts."
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_physics_windows.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows/statistical_features_histgb"),
    )
    parser.add_argument(
        "--model",
        choices=["hist_gb", "extra_trees", "random_forest"],
        default="hist_gb",
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument(
        "--threshold-strategy",
        choices=["f1_calibration", "accuracy_calibration", "normal_quantile"],
        default="f1_calibration",
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    data = np.load(args.windows, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y_multiclass = data["y"].astype(np.int64)
    y = (y_multiclass != 0).astype(np.int64)
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]
    feature_names = np.asarray([str(item) for item in data["feature_names"]])

    x_features, stat_feature_names = statistical_features(x, feature_names)
    train_idx, cal_idx, test_idx = grouped_train_cal_test_split(
        y=y,
        files=files,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
    )

    print(
        f"Dataset windows={len(y)}, stat_features={x_features.shape[1]}, "
        f"normal={int((y == 0).sum())}, anomaly={int((y == 1).sum())}"
    )
    print_split_summary("train", y[train_idx])
    print_split_summary("calibration", y[cal_idx])
    print_split_summary("test", y[test_idx])

    model = build_model(args.model, args.seed)
    model.fit(x_features[train_idx], y[train_idx])

    cal_scores = score_model(model, x_features[cal_idx])
    test_scores = score_model(model, x_features[test_idx])
    if args.threshold_strategy == "f1_calibration":
        threshold = metric_optimal_threshold(cal_scores, y[cal_idx], metric="f1")
    elif args.threshold_strategy == "accuracy_calibration":
        threshold = metric_optimal_threshold(cal_scores, y[cal_idx], metric="accuracy")
    else:
        threshold = normal_quantile_threshold(cal_scores, y[cal_idx], args.threshold_quantile)

    evaluation = evaluate(
        y_true=y[test_idx],
        scores=test_scores,
        threshold=threshold,
        labels=labels[test_idx],
        files=files[test_idx],
        starts=starts[test_idx],
        normal_threshold_quantile=args.threshold_quantile,
    )
    evaluation["model"] = f"Statistical window features + {args.model}"
    evaluation["threshold"] = threshold
    evaluation["threshold_strategy"] = args.threshold_strategy
    evaluation["interpretation"] = "anomaly_probability > threshold is anomalous"
    evaluation["train"] = split_summary(y[train_idx])
    evaluation["calibration"] = split_summary(y[cal_idx])
    evaluation["test"]["windows"] = int(len(test_idx))
    evaluation["hyperparameters"] = {
        "windows": str(args.windows),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "seed": args.seed,
        "split": {
            "test_size": args.test_size,
            "calibration_size": args.calibration_size,
            "test_random_state": 11,
            "calibration_random_state": 13,
            "grouped_by": "files",
        },
        "threshold": {
            "strategy": args.threshold_strategy,
            "normal_quantile": args.threshold_quantile,
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
            "stat_feature_names": stat_feature_names,
            "source_feature_names": feature_names.tolist(),
        },
        args.output_dir / "statistical_feature_model.joblib",
    )
    (args.output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(evaluation["hyperparameters"], indent=2),
        encoding="utf-8",
    )
    write_predictions(args.output_dir / "predictions.csv", evaluation["predictions"])

    print(f"\nWrote model to {args.output_dir / 'statistical_feature_model.joblib'}")
    print_result(threshold, evaluation)


def statistical_features(x: np.ndarray, feature_names: np.ndarray) -> tuple[np.ndarray, list[str]]:
    stats = {
        "mean": np.mean(x, axis=2),
        "std": np.std(x, axis=2),
        "min": np.min(x, axis=2),
        "max": np.max(x, axis=2),
        "last_minus_first": x[:, :, -1] - x[:, :, 0],
        "q10": np.quantile(x, 0.10, axis=2),
        "q90": np.quantile(x, 0.90, axis=2),
        "energy": np.mean(x * x, axis=2),
    }
    names: list[str] = []
    matrices: list[np.ndarray] = []
    for stat_name, values in stats.items():
        matrices.append(values.astype(np.float32))
        names.extend([f"{feature}.{stat_name}" for feature in feature_names.tolist()])
    return np.concatenate(matrices, axis=1), names


def build_model(model_name: str, seed: int):
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=1000,
            class_weight="balanced",
            max_features=0.4,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=seed,
    )


def grouped_train_cal_test_split(
    y: np.ndarray,
    files: np.ndarray,
    test_size: float,
    calibration_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    first_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=11)
    train_cal_idx, test_idx = next(first_split.split(indices, y, groups=files))

    second_split = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=13)
    train_rel, cal_rel = next(second_split.split(train_cal_idx, y[train_cal_idx], groups=files[train_cal_idx]))
    return train_cal_idx[train_rel], train_cal_idx[cal_rel], test_idx


def score_model(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def metric_optimal_threshold(scores: np.ndarray, y: np.ndarray, metric: str) -> float:
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in threshold_candidates(scores):
        y_pred = (scores > threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y,
            y_pred,
            average="binary",
            zero_division=0,
        )
        accuracy = float(accuracy_score(y, y_pred))
        key = (float(f1), accuracy, float(recall)) if metric == "f1" else (accuracy, float(f1), float(recall))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def normal_quantile_threshold(scores: np.ndarray, y: np.ndarray, quantile: float) -> float:
    normal_scores = scores[y == 0]
    if normal_scores.size == 0:
        return float(np.quantile(scores, quantile))
    return float(np.quantile(normal_scores, quantile))


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return np.asarray([0.5])
    return np.r_[unique_scores.min() - 1e-9, unique_scores, unique_scores.max() + 1e-9]


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    files: np.ndarray,
    starts: np.ndarray,
    normal_threshold_quantile: float,
) -> dict[str, Any]:
    y_pred = (scores > threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = 0.0

    normal_mask = y_true == 0
    anomaly_mask = y_true == 1
    predictions = [
        {
            "label": str(label),
            "y_true": int(true),
            "file": str(file),
            "window_start_s": float(start),
            "anomaly_probability": float(score),
            "flagged": int(flagged),
        }
        for label, true, file, start, score, flagged in zip(labels, y_true, files, starts, scores, y_pred)
    ]

    by_label: dict[str, dict[str, float]] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        label_scores = scores[mask]
        label_pred = y_pred[mask]
        by_label[str(label)] = {
            "n": int(mask.sum()),
            "mean_score": float(label_scores.mean()) if label_scores.size else 0.0,
            "flagged_count": int(label_pred.sum()),
            "flagged_rate": float(label_pred.mean()) if label_pred.size else 0.0,
        }

    return {
        "test": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "normal_flag_rate": float(np.mean(y_pred[normal_mask])) if np.any(normal_mask) else 0.0,
            "anomaly_flag_rate": float(np.mean(y_pred[anomaly_mask])) if np.any(anomaly_mask) else 0.0,
            "normal_threshold_quantile": normal_threshold_quantile,
        },
        "by_label": by_label,
        "predictions": predictions,
    }


def split_summary(y: np.ndarray) -> dict[str, int]:
    return {
        "windows": int(len(y)),
        "normal": int((y == 0).sum()),
        "anomaly": int((y == 1).sum()),
    }


def print_split_summary(name: str, y: np.ndarray) -> None:
    summary = split_summary(y)
    print(f"{name.title()} split: windows={summary['windows']}, normal={summary['normal']}, anomaly={summary['anomaly']}")


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = ["label", "y_true", "file", "window_start_s", "anomaly_probability", "flagged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def print_result(threshold: float, evaluation: dict[str, Any]) -> None:
    test = evaluation["test"]
    print(f"Threshold: {threshold:.6f}")
    print(f"Test ROC-AUC: {test['roc_auc']:.4f}")
    print(f"Test accuracy: {test['accuracy']:.4f}")
    print(f"Test precision: {test['precision']:.4f}")
    print(f"Test recall: {test['recall']:.4f}")
    print(f"Test F1: {test['f1']:.4f}")
    print(f"Test normal flag rate: {test['normal_flag_rate']:.2%}")
    print(f"Test anomaly flag rate: {test['anomaly_flag_rate']:.2%}")
    for label, stats in evaluation["by_label"].items():
        print(
            f"  {label}: n={stats['n']}, "
            f"flagged={stats['flagged_rate']:.2%}, mean_score={stats['mean_score']:.4f}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_family_thresholds import aligned_probabilities, predict_with_family_thresholds, safe_name
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
            "Search 1000 per-family UAV-SEAD threshold configurations from a fixed grid, "
            "select on calibration data, and evaluate once on the held-out test split."
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
        default=Path("data/uav_sead/moment_windows/statistical_features_family_threshold_search_1000"),
    )
    parser.add_argument(
        "--old-eval",
        type=Path,
        default=Path("data/uav_sead/moment_windows/statistical_features_histgb/evaluation.json"),
        help="Previous best evaluation.json to compare against.",
    )
    parser.add_argument("--model", choices=["hist_gb", "extra_trees", "random_forest"], default="hist_gb")
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "f1", "balanced"],
        default="accuracy",
        help="Calibration metric used to choose the winning threshold configuration.",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        raise SystemExit("--trials must be positive.")
    if not 0.0 < args.threshold_step <= 1.0:
        raise SystemExit("--threshold-step must be in (0, 1].")

    data = np.load(args.windows, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    y_binary = (y != 0).astype(np.int64)
    labels = np.asarray([str(item) for item in data["labels"]])
    files = data["files"]
    starts = data["window_start_s"]
    class_names = np.asarray([str(item) for item in data["class_names"]])
    feature_names = np.asarray([str(item) for item in data["feature_names"]])
    anomaly_class_ids = [int(class_id) for class_id in sorted(np.unique(y)) if int(class_id) != 0]

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

    threshold_configs = generate_threshold_configs(
        class_ids=anomaly_class_ids,
        step=args.threshold_step,
        trials=args.trials,
        seed=args.seed,
    )
    search_rows = run_search(
        threshold_configs=threshold_configs,
        probabilities=cal_prob,
        y_true=y[cal_idx],
        class_ids=anomaly_class_ids,
        class_names=class_names,
    )
    best_row = select_best(search_rows, args.selection_metric)
    thresholds = thresholds_from_row(best_row, anomaly_class_ids, class_names)

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
    evaluation["model"] = f"Statistical window features + {args.model} + 1000-config per-family threshold search"
    evaluation["threshold_strategy"] = f"per_family_{args.selection_metric}_1000_config_grid"
    evaluation["threshold"] = {class_names[class_id]: thresholds[class_id]["threshold"] for class_id in anomaly_class_ids}
    evaluation["interpretation"] = (
        "1000 threshold configurations are scored on calibration data; "
        "the selected per-family thresholds are evaluated once on the held-out test split."
    )
    evaluation["train"] = split_summary(y_binary[train_idx])
    evaluation["calibration"] = split_summary(y_binary[cal_idx])
    evaluation["test"]["windows"] = int(len(test_idx))
    evaluation["threshold_search"] = {
        "trials_requested": args.trials,
        "trials_run": len(search_rows),
        "threshold_step": args.threshold_step,
        "selection_metric": args.selection_metric,
        "best_trial": int(best_row["trial"]),
        "best_calibration_accuracy": float(best_row["calibration_accuracy"]),
        "best_calibration_precision": float(best_row["calibration_precision"]),
        "best_calibration_recall": float(best_row["calibration_recall"]),
        "best_calibration_f1": float(best_row["calibration_f1"]),
    }
    evaluation["hyperparameters"] = {
        "windows": str(args.windows),
        "output_dir": str(args.output_dir),
        "old_eval": str(args.old_eval),
        "model": args.model,
        "seed": args.seed,
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
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "thresholds": thresholds,
            "class_names": class_names.tolist(),
            "stat_feature_names": stat_feature_names,
            "source_feature_names": feature_names.tolist(),
        },
        args.output_dir / "family_threshold_search_model.joblib",
    )
    (args.output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(evaluation["hyperparameters"], indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "best_thresholds.json").write_text(
        json.dumps({class_names[class_id]: thresholds[class_id] for class_id in anomaly_class_ids}, indent=2),
        encoding="utf-8",
    )
    write_search_csv(args.output_dir / "threshold_search.csv", search_rows, anomaly_class_ids, class_names)
    write_predictions(args.output_dir / "predictions.csv", evaluation["predictions"], class_names)
    comparison_rows = build_comparison_rows(args.old_eval, evaluation)
    write_comparison_csv(args.output_dir / "comparison_vs_old_best.csv", comparison_rows)
    write_figures(
        figures_dir=figures_dir,
        search_rows=search_rows,
        best_row=best_row,
        comparison_rows=comparison_rows,
        evaluation=evaluation,
        class_names=class_names,
        anomaly_class_ids=anomaly_class_ids,
    )

    print(f"\nWrote model to {args.output_dir / 'family_threshold_search_model.joblib'}")
    print_result(evaluation, class_names, anomaly_class_ids)
    print("\n1000-search comparison:")
    for row in comparison_rows:
        print(
            f"  {row['run']}: accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}, roc_auc={row['roc_auc']:.4f}"
        )
    print(f"\nCharts written to {figures_dir}")


def generate_threshold_configs(class_ids: list[int], step: float, trials: int, seed: int) -> list[tuple[float, ...]]:
    grid = [float(value) for value in np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)]
    total_configs = len(grid) ** len(class_ids)
    if trials > total_configs:
        raise SystemExit(
            f"Requested {trials} trials, but the {step} threshold grid only has {total_configs} unique configs."
        )

    seeded_configs = [
        tuple([0.05] * len(class_ids)),
        tuple([0.10] * len(class_ids)),
        tuple([0.15] * len(class_ids)),
        tuple([0.20] * len(class_ids)),
    ]
    if len(class_ids) == 4:
        seeded_configs.insert(0, (0.15, 0.05, 0.10, 0.55))

    rng = random.Random(seed)
    configs: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for config in seeded_configs:
        rounded = tuple(round(value, 10) for value in config)
        if rounded not in seen:
            configs.append(rounded)
            seen.add(rounded)

    while len(configs) < trials:
        config = tuple(float(rng.choice(grid)) for _ in class_ids)
        if config not in seen:
            configs.append(config)
            seen.add(config)
    return configs


def run_search(
    threshold_configs: list[tuple[float, ...]],
    probabilities: np.ndarray,
    y_true: np.ndarray,
    class_ids: list[int],
    class_names: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_binary_true = (y_true != 0).astype(np.int64)
    for trial, config in enumerate(threshold_configs, start=1):
        thresholds = {
            class_id: {
                "class_id": int(class_id),
                "class_name": str(class_names[class_id]),
                "threshold": float(config[index]),
            }
            for index, class_id in enumerate(class_ids)
        }
        y_pred = predict_with_family_thresholds(probabilities, thresholds, class_ids)
        y_binary_pred = (y_pred != 0).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_binary_true,
            y_binary_pred,
            average="binary",
            zero_division=0,
        )
        normal_mask = y_binary_true == 0
        anomaly_mask = y_binary_true == 1
        row: dict[str, Any] = {
            "trial": trial,
            "calibration_accuracy": float(accuracy_score(y_binary_true, y_binary_pred)),
            "calibration_precision": float(precision),
            "calibration_recall": float(recall),
            "calibration_f1": float(f1),
            "calibration_multiclass_accuracy": float(accuracy_score(y_true, y_pred)),
            "calibration_normal_flag_rate": float(np.mean(y_binary_pred[normal_mask])) if np.any(normal_mask) else 0.0,
            "calibration_anomaly_flag_rate": float(np.mean(y_binary_pred[anomaly_mask])) if np.any(anomaly_mask) else 0.0,
        }
        for index, class_id in enumerate(class_ids):
            row[f"threshold_{safe_name(class_names[class_id])}"] = float(config[index])
        rows.append(row)
    return rows


def select_best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    best_row = rows[0]
    best_key = search_key(best_row, metric)
    for row in rows[1:]:
        key = search_key(row, metric)
        if key > best_key:
            best_key = key
            best_row = row
    return best_row


def search_key(row: dict[str, Any], metric: str) -> tuple[float, float, float]:
    if metric == "f1":
        return (float(row["calibration_f1"]), float(row["calibration_accuracy"]), float(row["calibration_recall"]))
    if metric == "balanced":
        normal_specificity = 1.0 - float(row["calibration_normal_flag_rate"])
        balanced = (normal_specificity + float(row["calibration_anomaly_flag_rate"])) / 2.0
        return (balanced, float(row["calibration_f1"]), float(row["calibration_accuracy"]))
    return (float(row["calibration_accuracy"]), float(row["calibration_f1"]), float(row["calibration_recall"]))


def thresholds_from_row(
    row: dict[str, Any],
    class_ids: list[int],
    class_names: np.ndarray,
) -> dict[int, dict[str, Any]]:
    return {
        class_id: {
            "class_id": int(class_id),
            "class_name": str(class_names[class_id]),
            "threshold": float(row[f"threshold_{safe_name(class_names[class_id])}"]),
            "selected_trial": int(row["trial"]),
            "selection_calibration_accuracy": float(row["calibration_accuracy"]),
            "selection_calibration_precision": float(row["calibration_precision"]),
            "selection_calibration_recall": float(row["calibration_recall"]),
            "selection_calibration_f1": float(row["calibration_f1"]),
        }
        for class_id in class_ids
    }


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
    by_label = build_by_label(labels, y_binary_pred, anomaly_score)
    per_family = build_per_family(y_true, y_pred, y_binary_true, class_names, thresholds, anomaly_class_ids)
    predictions = build_predictions(labels, y_true, y_pred, files, starts, probabilities, anomaly_score, y_binary_pred, class_names)

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


def build_by_label(labels: np.ndarray, y_binary_pred: np.ndarray, anomaly_score: np.ndarray) -> dict[str, dict[str, float]]:
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
    return by_label


def build_per_family(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_binary_true: np.ndarray,
    class_names: np.ndarray,
    thresholds: dict[int, dict[str, Any]],
    anomaly_class_ids: list[int],
) -> dict[str, dict[str, float]]:
    normal_mask = y_binary_true == 0
    per_family: dict[str, dict[str, float]] = {}
    for class_id in anomaly_class_ids:
        class_mask = y_true == class_id
        predicted_mask = y_pred == class_id
        non_family_mask = y_true != class_id
        precision, recall, f1, _ = precision_recall_fscore_support(
            class_mask.astype(np.int64),
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
            "test_normal_false_family_rate": float(np.mean(predicted_mask[normal_mask])) if np.any(normal_mask) else 0.0,
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
        }
    return per_family


def build_predictions(
    labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    files: np.ndarray,
    starts: np.ndarray,
    probabilities: np.ndarray,
    anomaly_score: np.ndarray,
    y_binary_pred: np.ndarray,
    class_names: np.ndarray,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
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
    return predictions


def build_comparison_rows(old_eval_path: Path, evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if old_eval_path.exists():
        old_payload = json.loads(old_eval_path.read_text(encoding="utf-8"))
        rows.append({"run": old_eval_path.parent.name, **metric_subset(old_payload["test"])})
    rows.append({"run": "family_threshold_search_1000", **metric_subset(evaluation["test"])})
    return rows


def metric_subset(test: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(test.get("accuracy", 0.0)),
        "precision": float(test.get("precision", 0.0)),
        "recall": float(test.get("recall", 0.0)),
        "f1": float(test.get("f1", 0.0)),
        "roc_auc": float(test.get("roc_auc", 0.0)),
        "normal_flag_rate": float(test.get("normal_flag_rate", 0.0)),
        "anomaly_flag_rate": float(test.get("anomaly_flag_rate", 0.0)),
    }


def write_search_csv(path: Path, rows: list[dict[str, Any]], class_ids: list[int], class_names: np.ndarray) -> None:
    fields = [
        "trial",
        *[f"threshold_{safe_name(class_names[class_id])}" for class_id in class_ids],
        "calibration_accuracy",
        "calibration_precision",
        "calibration_recall",
        "calibration_f1",
        "calibration_multiclass_accuracy",
        "calibration_normal_flag_rate",
        "calibration_anomaly_flag_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


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


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["run", "accuracy", "precision", "recall", "f1", "roc_auc", "normal_flag_rate", "anomaly_flag_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(
    figures_dir: Path,
    search_rows: list[dict[str, Any]],
    best_row: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    evaluation: dict[str, Any],
    class_names: np.ndarray,
    anomaly_class_ids: list[int],
) -> None:
    write_search_figure(figures_dir, search_rows, best_row)
    write_comparison_figure(figures_dir, comparison_rows)
    write_threshold_figure(figures_dir, evaluation, class_names, anomaly_class_ids)
    write_family_detection_figure(figures_dir, evaluation)
    write_confusion_figure(figures_dir, evaluation, class_names)


def write_search_figure(figures_dir: Path, rows: list[dict[str, Any]], best_row: dict[str, Any]) -> None:
    trials = [int(row["trial"]) for row in rows]
    accuracy = [float(row["calibration_accuracy"]) for row in rows]
    f1 = [float(row["calibration_f1"]) for row in rows]
    best_trial = int(best_row["trial"])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(trials, accuracy, color="#2f6fbb", linewidth=1.4, label="Calibration accuracy")
    ax.plot(trials, f1, color="#4f9d69", linewidth=1.4, label="Calibration F1")
    ax.axvline(best_trial, color="#c4473d", linestyle="--", linewidth=1.2, label=f"Selected trial {best_trial}")
    ax.set_xlabel("Threshold Configuration Trial")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("1000 Per-Family Threshold Configurations")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    save_figure(fig, figures_dir / "threshold-search-1000")


def write_comparison_figure(figures_dir: Path, rows: list[dict[str, Any]]) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#777777", "#2f6fbb"]
    for offset, row in enumerate(rows):
        values = [float(row[metric]) for metric in metrics]
        positions = x + (offset - (len(rows) - 1) / 2.0) * width
        bars = ax.bar(positions, values, width=width, label=row["run"], color=colors[offset % len(colors)])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Old Best vs 1000-Config Family Threshold Search")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    save_figure(fig, figures_dir / "comparison-vs-old-best")


def write_threshold_figure(figures_dir: Path, evaluation: dict[str, Any], class_names: np.ndarray, class_ids: list[int]) -> None:
    names = [str(class_names[class_id]) for class_id in class_ids]
    values = [float(evaluation["per_family"][name]["threshold"]) for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(names, values, color="#2f6fbb")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Threshold")
    ax.set_title("Selected Threshold By Anomaly Family")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", va="bottom")
    save_figure(fig, figures_dir / "selected-family-thresholds")


def write_family_detection_figure(figures_dir: Path, evaluation: dict[str, Any]) -> None:
    names = list(evaluation["per_family"])
    values = [float(evaluation["per_family"][name]["test_detection_rate"]) for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(names, values, color="#4f9d69")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection Rate")
    ax.set_title("Held-Out Detection Rate By Anomaly Family")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center", va="bottom")
    save_figure(fig, figures_dir / "family-detection-rate")


def write_confusion_figure(figures_dir: Path, evaluation: dict[str, Any], class_names: np.ndarray) -> None:
    y_true = np.asarray([int(row["y_true_class"]) for row in evaluation["predictions"]])
    y_pred = np.asarray([int(row["predicted_class"]) for row in evaluation["predictions"]])
    labels = [int(class_id) for class_id in sorted(set(y_true.tolist()) | set(y_pred.tolist()))]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    names = [str(class_names[class_id]) for class_id in labels]

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.set_title("Multiclass Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > matrix.max() * 0.55 else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, figures_dir / "multiclass-confusion-matrix")


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=220)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    plt.close(fig)


def print_result(evaluation: dict[str, Any], class_names: np.ndarray, anomaly_class_ids: list[int]) -> None:
    print("Selected thresholds:")
    for class_id in anomaly_class_ids:
        name = str(class_names[class_id])
        stats = evaluation["per_family"][name]
        print(
            f"  {name}: threshold={stats['threshold']:.2f}, "
            f"detection={stats['test_detection_rate']:.2%}, "
            f"precision={stats['test_precision']:.4f}, f1={stats['test_f1']:.4f}"
        )
    test = evaluation["test"]
    print("Held-out test metrics:")
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

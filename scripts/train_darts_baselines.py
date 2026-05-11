from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from darts import TimeSeries
from darts.ad.scorers import KMeansScorer, PyODScorer
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


MODEL_CHOICES = ("kmeans", "pyod_ecod", "pyod_iforest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Darts anomaly-scoring baselines on UAV-SEAD window datasets."
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_windows.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows"),
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--threshold-strategy",
        choices=["normal_quantile", "f1_calibration"],
        default="f1_calibration",
    )
    parser.add_argument("--window", type=int, default=8, help="Darts rolling scoring window.")
    parser.add_argument("--k", type=int, default=8, help="KMeans cluster count.")
    parser.add_argument(
        "--score-agg",
        choices=["mean", "max", "p95"],
        default="p95",
        help="How to collapse per-timestep Darts scores into one score per UAV window.",
    )
    parser.add_argument(
        "--max-train-normal",
        type=int,
        default=1200,
        help="Cap normal training windows used to fit Darts scorers. 0 uses all normal train windows.",
    )
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--limit-windows", type=int, default=0, help="Optional quick smoke-test limit.")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    data = np.load(args.windows, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y_multiclass = data["y"].astype(np.int64)
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]
    y = (y_multiclass != 0).astype(np.int64)

    if args.limit_windows > 0:
        limit = min(args.limit_windows, x.shape[0])
        x = x[:limit]
        y = y[:limit]
        labels = labels[:limit]
        files = files[:limit]
        starts = starts[:limit]

    train_idx, cal_idx, test_idx = grouped_train_cal_test_split(
        x=x,
        y=y,
        files=files,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
    )
    train_normal_idx = train_idx[y[train_idx] == 0]
    if train_normal_idx.size == 0:
        raise SystemExit("Darts baselines require at least one normal training window.")
    if args.max_train_normal > 0 and train_normal_idx.size > args.max_train_normal:
        rng = np.random.default_rng(args.seed)
        train_normal_idx = np.sort(
            rng.choice(train_normal_idx, size=args.max_train_normal, replace=False)
        )

    series_cache: dict[int, TimeSeries] = {}

    def get_series(indices: np.ndarray) -> list[TimeSeries]:
        series = []
        for index in indices.tolist():
            if index not in series_cache:
                series_cache[index] = TimeSeries.from_values(x[index].T)
            series.append(series_cache[index])
        return series

    for model_name in args.models:
        experiment_dir = args.output_dir / f"darts_{model_name}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        scorer = build_scorer(model_name, args)
        print(
            f"Fitting Darts {model_name} on {len(train_normal_idx)} normal windows "
            f"(window={args.window}, score_agg={args.score_agg})",
            flush=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scorer.fit(get_series(train_normal_idx))

        cal_scores = score_indices(
            scorer=scorer,
            indices=cal_idx,
            get_series=get_series,
            batch_size=args.score_batch_size,
            aggregate=args.score_agg,
        )
        test_scores = score_indices(
            scorer=scorer,
            indices=test_idx,
            get_series=get_series,
            batch_size=args.score_batch_size,
            aggregate=args.score_agg,
        )

        if args.threshold_strategy == "f1_calibration":
            threshold = f1_optimal_threshold(cal_scores, y[cal_idx])
        else:
            threshold = normal_quantile_threshold(cal_scores, y[cal_idx], args.threshold_quantile)

        eval_payload = evaluate(
            y_true=y[test_idx],
            scores=test_scores,
            threshold=threshold,
            labels=labels[test_idx],
            files=files[test_idx],
            starts=starts[test_idx],
            normal_threshold_quantile=args.threshold_quantile,
        )
        eval_payload["train"] = split_summary(y[train_idx])
        eval_payload["calibration"] = split_summary(y[cal_idx])
        eval_payload["test"]["windows"] = int(len(test_idx))
        eval_payload["model"] = f"Darts {model_name}"
        eval_payload["threshold"] = threshold
        eval_payload["threshold_strategy"] = args.threshold_strategy
        eval_payload["score_aggregation"] = args.score_agg
        eval_payload["darts_window"] = args.window
        eval_payload["train_normal_windows_used"] = int(len(train_normal_idx))
        eval_payload["interpretation"] = "darts_anomaly_score > threshold is anomalous"

        (experiment_dir / "evaluation.json").write_text(
            json.dumps(eval_payload, indent=2),
            encoding="utf-8",
        )
        write_predictions(experiment_dir / "predictions.csv", eval_payload["predictions"])
        try:
            joblib.dump(scorer, experiment_dir / "darts_scorer.joblib")
        except Exception as exc:  # pragma: no cover - serialization differs by scorer.
            (experiment_dir / "scorer_save_error.txt").write_text(str(exc), encoding="utf-8")

        print_result(model_name, threshold, eval_payload)


def build_scorer(model_name: str, args: argparse.Namespace):
    if model_name == "kmeans":
        return KMeansScorer(
            window=args.window,
            k=args.k,
            component_wise=False,
            window_agg=True,
            random_state=args.seed,
        )
    if model_name == "pyod_ecod":
        return PyODScorer(
            ECOD(),
            window=args.window,
            component_wise=False,
            window_agg=True,
        )
    if model_name == "pyod_iforest":
        return PyODScorer(
            IForest(
                n_estimators=300,
                contamination=0.10,
                random_state=args.seed,
                n_jobs=-1,
            ),
            window=args.window,
            component_wise=False,
            window_agg=True,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def score_indices(
    scorer,
    indices: np.ndarray,
    get_series: Callable[[np.ndarray], list[TimeSeries]],
    batch_size: int,
    aggregate: str,
) -> np.ndarray:
    scores: list[float] = []
    total = len(indices)
    for start in range(0, total, batch_size):
        batch_idx = indices[start : start + batch_size]
        score_series = scorer.score(get_series(batch_idx))
        if isinstance(score_series, TimeSeries):
            score_series = [score_series]
        scores.extend(aggregate_score(series, aggregate) for series in score_series)
        print(f"  scored {min(start + batch_size, total)}/{total}", flush=True)
    return np.asarray(scores, dtype=np.float64)


def aggregate_score(series: TimeSeries, aggregate: str) -> float:
    values = series.values(copy=False).astype(np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if aggregate == "max":
        return float(np.nanmax(values))
    if aggregate == "p95":
        return float(np.nanquantile(values, 0.95))
    return float(np.nanmean(values))


def grouped_train_cal_test_split(
    x: np.ndarray,
    y: np.ndarray,
    files: np.ndarray,
    test_size: float,
    calibration_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=11)
    train_cal_idx, test_idx = next(splitter.split(x, y, groups=files))
    cal_splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=13)
    train_rel_idx, cal_rel_idx = next(
        cal_splitter.split(
            x[train_cal_idx],
            y[train_cal_idx],
            groups=files[train_cal_idx],
        )
    )
    train_idx = train_cal_idx[train_rel_idx]
    cal_idx = train_cal_idx[cal_rel_idx]
    return train_idx, cal_idx, test_idx


def normal_quantile_threshold(scores: np.ndarray, y: np.ndarray, quantile: float) -> float:
    normal_scores = scores[y == 0]
    if normal_scores.size == 0:
        return float(np.quantile(scores, quantile))
    return float(np.quantile(normal_scores, quantile))


def f1_optimal_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    candidates = threshold_candidates(scores)
    if candidates.size == 0:
        return 0.0

    best_threshold = float(candidates[0])
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        y_pred = (scores > threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y,
            y_pred,
            average="binary",
            zero_division=0,
        )
        key = (float(f1), float(recall), float(precision))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return np.asarray([], dtype=np.float64)

    eps = 1e-12
    if unique_scores.size == 1:
        return np.asarray([unique_scores[0] - eps, unique_scores[0] + eps])
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    return np.concatenate(([unique_scores[0] - eps], midpoints, [unique_scores[-1] + eps]))


def split_summary(y: np.ndarray) -> dict[str, int]:
    return {
        "windows": int(y.size),
        "normal_windows": int((y == 0).sum()),
        "anomaly_windows": int((y == 1).sum()),
    }


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
    normal_flag_rate = float(np.mean(y_pred[normal_mask])) if np.any(normal_mask) else 0.0
    anomaly_flag_rate = float(np.mean(y_pred[anomaly_mask])) if np.any(anomaly_mask) else 0.0

    predictions = [
        {
            "label": str(label),
            "y_true": int(true),
            "file": str(file),
            "window_start_s": float(start),
            "darts_anomaly_score": float(score),
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
            "mean_score": float(np.mean(label_scores)) if label_scores.size else 0.0,
            "p95_score": float(np.quantile(label_scores, 0.95)) if label_scores.size else 0.0,
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
            "normal_flag_rate": normal_flag_rate,
            "anomaly_flag_rate": anomaly_flag_rate,
            "normal_threshold_quantile": normal_threshold_quantile,
        },
        "by_label": by_label,
        "predictions": predictions,
    }


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = ["label", "y_true", "file", "window_start_s", "darts_anomaly_score", "flagged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def print_result(model_name: str, threshold: float, eval_payload: dict[str, Any]) -> None:
    test = eval_payload["test"]
    print(f"\nDarts {model_name}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Test ROC-AUC: {test['roc_auc']:.4f}")
    print(f"Test accuracy: {test['accuracy']:.4f}")
    print(f"Test precision: {test['precision']:.4f}")
    print(f"Test recall: {test['recall']:.4f}")
    print(f"Test F1: {test['f1']:.4f}")
    print(f"Test normal flag rate: {test['normal_flag_rate']:.2%}")
    print(f"Test anomaly flag rate: {test['anomaly_flag_rate']:.2%}")
    for label, stats in eval_payload["by_label"].items():
        print(f"  {label}: n={stats['n']}, flagged={stats['flagged_rate']:.2%}, mean_score={stats['mean_score']:.4f}")


if __name__ == "__main__":
    main()

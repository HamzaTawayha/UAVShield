from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from momentfm import MOMENTPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    run_start = time.time()
    parser = argparse.ArgumentParser(
        description="Train a lightweight anomaly head on top of pretrained MOMENT embeddings."
    )
    parser.add_argument("--windows", type=Path, default=Path("data/px4_flight_review/moment_windows/px4_moment_windows.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/px4_flight_review/moment_windows/moment_binary_head"))
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--threshold-strategy",
        choices=["normal_quantile", "f1_calibration"],
        default="normal_quantile",
        help=(
            "normal_quantile limits false positives using normal calibration scores; "
            "f1_calibration chooses the threshold with best calibration F1."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head", choices=["logistic", "random_forest"], default="random_forest")
    parser.add_argument(
        "--sklearn-verbose",
        type=int,
        default=1,
        help="Verbosity passed into the sklearn head. Use 0 for quiet sklearn fitting.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for MOMENT embedding. auto uses CUDA when PyTorch can see it.",
    )
    args = parser.parse_args()

    checkpoint("parsed arguments", run_start)
    hyperparameters = build_hyperparameters(args)
    print_hyperparameters(hyperparameters)

    checkpoint(f"loading windows from {args.windows}", run_start)
    data = np.load(args.windows, allow_pickle=True)
    x = data["x"]
    y_multiclass = data["y"]
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]
    y = (y_multiclass != 0).astype(np.int64)
    print(
        "Dataset summary: "
        f"windows={x.shape[0]}, channels={x.shape[1]}, seq_len={x.shape[2]}, "
        f"normal={int((y == 0).sum())}, anomaly={int((y == 1).sum())}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(hyperparameters, indent=2),
        encoding="utf-8",
    )
    checkpoint(f"wrote run config to {args.output_dir / 'run_config.json'}", run_start)

    embeddings_path = args.output_dir / "moment_embeddings.npz"
    if embeddings_path.exists():
        checkpoint(f"loading cached MOMENT embeddings from {embeddings_path}", run_start)
        embeddings = np.load(embeddings_path, allow_pickle=True)["embeddings"]
        print(f"Loaded cached embeddings from {embeddings_path}: shape={embeddings.shape}", flush=True)
    else:
        checkpoint("building MOMENT embeddings", run_start)
        embeddings = build_moment_embeddings(x, batch_size=args.batch_size, device_request=args.device)
        checkpoint(f"saving MOMENT embeddings to {embeddings_path}", run_start)
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings.astype(np.float32),
            y=y,
            labels=labels,
            files=files,
            window_start_s=starts,
        )

    checkpoint("creating grouped train/calibration/test split", run_start)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=11)
    train_cal_idx, test_idx = next(splitter.split(embeddings, y, groups=files))
    cal_splitter = GroupShuffleSplit(n_splits=1, test_size=args.calibration_size, random_state=13)
    train_rel_idx, cal_rel_idx = next(
        cal_splitter.split(
            embeddings[train_cal_idx],
            y[train_cal_idx],
            groups=files[train_cal_idx],
        )
    )
    train_idx = train_cal_idx[train_rel_idx]
    cal_idx = train_cal_idx[cal_rel_idx]
    print_split_summary("train", y[train_idx])
    print_split_summary("calibration", y[cal_idx])
    print_split_summary("test", y[test_idx])

    clf = build_head(args.head, verbose=args.sklearn_verbose)
    checkpoint(f"fitting {args.head} head on {len(train_idx)} windows", run_start)
    clf.fit(embeddings[train_idx], y[train_idx])
    checkpoint("head fit complete", run_start)

    checkpoint("scoring calibration split", run_start)
    cal_scores = clf.predict_proba(embeddings[cal_idx])[:, 1]
    checkpoint("scoring test split", run_start)
    test_scores = clf.predict_proba(embeddings[test_idx])[:, 1]
    if args.threshold_strategy == "f1_calibration":
        checkpoint("choosing threshold by calibration F1", run_start)
        threshold = f1_optimal_threshold(cal_scores, y[cal_idx])
    else:
        checkpoint(
            f"choosing threshold from normal calibration q={args.threshold_quantile}",
            run_start,
        )
        threshold = normal_quantile_threshold(cal_scores, y[cal_idx], args.threshold_quantile)

    checkpoint("evaluating test split", run_start)
    eval_payload = evaluate(
        y_true=y[test_idx],
        scores=test_scores,
        threshold=threshold,
        labels=labels[test_idx],
        files=files[test_idx],
        starts=starts[test_idx],
        train_normal_threshold_quantile=args.threshold_quantile,
    )
    eval_payload["train"] = {
        "windows": int(len(train_idx)),
        "normal_windows": int((y[train_idx] == 0).sum()),
        "anomaly_windows": int((y[train_idx] == 1).sum()),
    }
    eval_payload["calibration"] = {
        "windows": int(len(cal_idx)),
        "normal_windows": int((y[cal_idx] == 0).sum()),
        "anomaly_windows": int((y[cal_idx] == 1).sum()),
    }
    eval_payload["test"]["windows"] = int(len(test_idx))
    eval_payload["model"] = f"AutonLab/MOMENT-1-large embeddings + {args.head} head"
    eval_payload["threshold"] = threshold
    eval_payload["threshold_strategy"] = args.threshold_strategy
    eval_payload["interpretation"] = "anomaly_probability > threshold is anomalous"
    eval_payload["hyperparameters"] = hyperparameters

    checkpoint(f"saving model and evaluation artifacts to {args.output_dir}", run_start)
    joblib.dump(clf, args.output_dir / "moment_binary_head.joblib")
    (args.output_dir / "evaluation.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    write_predictions(args.output_dir / "predictions.csv", eval_payload["predictions"])

    print(f"Wrote model to {args.output_dir / 'moment_binary_head.joblib'}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Test ROC-AUC: {eval_payload['test']['roc_auc']:.4f}")
    print(f"Test accuracy: {eval_payload['test']['accuracy']:.4f}")
    print(f"Test precision: {eval_payload['test']['precision']:.4f}")
    print(f"Test recall: {eval_payload['test']['recall']:.4f}")
    print(f"Test F1: {eval_payload['test']['f1']:.4f}")
    print(f"Test normal flag rate: {eval_payload['test']['normal_flag_rate']:.2%}")
    print(f"Test anomaly flag rate: {eval_payload['test']['anomaly_flag_rate']:.2%}")
    for label, stats in eval_payload["by_label"].items():
        print(f"  {label}: n={stats['n']}, flagged={stats['flagged_rate']:.2%}, mean_score={stats['mean_score']:.4f}")
    checkpoint("run complete", run_start)


def build_head(name: str, verbose: int = 0):
    if name == "random_forest":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        random_state=11,
                        n_jobs=-1,
                        verbose=verbose,
                    ),
                ),
            ]
        )
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=11,
                    verbose=verbose,
                ),
            ),
        ]
    )


def build_hyperparameters(args: argparse.Namespace) -> dict:
    return {
        "windows": str(args.windows),
        "output_dir": str(args.output_dir),
        "moment": {
            "model": "AutonLab/MOMENT-1-large",
            "task_name": "embedding",
            "batch_size": args.batch_size,
            "device": args.device,
        },
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
        "head": head_hyperparameters(args.head, args.sklearn_verbose),
    }


def head_hyperparameters(name: str, verbose: int) -> dict:
    if name == "random_forest":
        return {
            "name": "random_forest",
            "scaler": "StandardScaler",
            "n_estimators": 400,
            "class_weight": "balanced_subsample",
            "min_samples_leaf": 2,
            "random_state": 11,
            "n_jobs": -1,
            "verbose": verbose,
        }
    return {
        "name": "logistic",
        "scaler": "StandardScaler",
        "class_weight": "balanced",
        "max_iter": 2000,
        "solver": "liblinear",
        "random_state": 11,
        "verbose": verbose,
    }


def print_hyperparameters(hyperparameters: dict) -> None:
    print("\nMOMENT training hyperparameters:", flush=True)
    print(json.dumps(hyperparameters, indent=2), flush=True)


def checkpoint(message: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"[checkpoint +{elapsed:.1f}s] {message}", flush=True)


def print_split_summary(name: str, y: np.ndarray) -> None:
    print(
        f"{name.capitalize()} split: windows={int(y.size)}, "
        f"normal={int((y == 0).sum())}, anomaly={int((y == 1).sum())}",
        flush=True,
    )


def build_moment_embeddings(x: np.ndarray, batch_size: int, device_request: str = "auto") -> np.ndarray:
    device = choose_device(device_request)
    total_batches = math.ceil(x.shape[0] / batch_size)
    print(
        f"Building MOMENT embeddings for {x.shape[0]} windows "
        f"({total_batches} batches, batch_size={batch_size}) on {device}",
        flush=True,
    )
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    start_time = time.time()
    with torch.no_grad():
        for batch_index, batch_start in enumerate(range(0, x.shape[0], batch_size), start=1):
            batch = torch.tensor(
                x[batch_start : batch_start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            output = model(x_enc=batch)
            embedding = getattr(output, "embeddings", None)
            if embedding is None:
                raise RuntimeError("MOMENT output did not expose embeddings")
            embeddings.append(embedding.detach().cpu().numpy())
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                elapsed = time.time() - start_time
                print(
                    f"  embedded batch {batch_index}/{total_batches} "
                    f"({batch_start + batch.shape[0]}/{x.shape[0]} windows, {elapsed:.1f}s)",
                    flush=True,
                )
    return np.vstack(embeddings)


def choose_device(requested: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a CUDA device. "
            "Check nvidia-smi, driver/container GPU access, and the installed torch build."
        )
    if requested == "cpu":
        return torch.device("cpu")
    if cuda_available:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda")
    print("CUDA is not available to PyTorch; falling back to CPU.", flush=True)
    return torch.device("cpu")


def normal_quantile_threshold(scores: np.ndarray, y: np.ndarray, quantile: float) -> float:
    normal_scores = scores[y == 0]
    if normal_scores.size == 0:
        return float(np.quantile(scores, quantile))
    return float(np.quantile(normal_scores, quantile))


def f1_optimal_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    candidates = threshold_candidates(scores)
    if candidates.size == 0:
        return 0.5

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


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    files: np.ndarray,
    starts: np.ndarray,
    train_normal_threshold_quantile: float,
) -> dict:
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
            "normal_threshold_quantile": train_normal_threshold_quantile,
        },
        "by_label": by_label,
        "predictions": predictions,
    }


def write_predictions(path: Path, predictions: list[dict]) -> None:
    fields = ["label", "y_true", "file", "window_start_s", "anomaly_probability", "flagged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    main()

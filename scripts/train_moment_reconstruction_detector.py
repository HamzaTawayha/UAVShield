from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from momentfm import MOMENTPipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a MOMENT reconstruction-error anomaly detector on windowed UAV logs."
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_state_windows.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows/moment_reconstruction_detector"),
    )
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    data = np.load(args.windows, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y_multiclass = data["y"].astype(np.int64)
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]
    y = (y_multiclass != 0).astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "moment_reconstruction_scores.npz"
    if scores_path.exists() and not args.force_recompute:
        scores = np.load(scores_path, allow_pickle=True)["scores"]
        print(f"Loaded cached reconstruction scores from {scores_path}")
    else:
        scores = build_reconstruction_scores(
            x,
            batch_size=args.batch_size,
            device_request=args.device,
        )
        np.savez_compressed(
            scores_path,
            scores=scores.astype(np.float32),
            y=y,
            labels=labels,
            files=files,
            window_start_s=starts,
        )

    train_idx, cal_idx, test_idx = grouped_train_cal_test_split(
        scores=scores,
        y=y,
        files=files,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
    )

    cal_scores = scores[cal_idx]
    if args.threshold_strategy == "f1_calibration":
        threshold = f1_optimal_threshold(cal_scores, y[cal_idx])
    else:
        threshold = normal_quantile_threshold(cal_scores, y[cal_idx], args.threshold_quantile)

    eval_payload = evaluate(
        y_true=y[test_idx],
        scores=scores[test_idx],
        threshold=threshold,
        labels=labels[test_idx],
        files=files[test_idx],
        starts=starts[test_idx],
        normal_threshold_quantile=args.threshold_quantile,
    )
    eval_payload["train"] = split_summary(y[train_idx])
    eval_payload["calibration"] = split_summary(y[cal_idx])
    eval_payload["test"]["windows"] = int(len(test_idx))
    eval_payload["model"] = "AutonLab/MOMENT-1-large reconstruction error"
    eval_payload["threshold"] = threshold
    eval_payload["threshold_strategy"] = args.threshold_strategy
    eval_payload["interpretation"] = "reconstruction_error > threshold is anomalous"

    (args.output_dir / "evaluation.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    write_predictions(args.output_dir / "predictions.csv", eval_payload["predictions"])

    print(f"Wrote evaluation to {args.output_dir / 'evaluation.json'}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Test ROC-AUC: {eval_payload['test']['roc_auc']:.4f}")
    print(f"Test accuracy: {eval_payload['test']['accuracy']:.4f}")
    print(f"Test precision: {eval_payload['test']['precision']:.4f}")
    print(f"Test recall: {eval_payload['test']['recall']:.4f}")
    print(f"Test F1: {eval_payload['test']['f1']:.4f}")
    print(f"Test normal flag rate: {eval_payload['test']['normal_flag_rate']:.2%}")
    print(f"Test anomaly flag rate: {eval_payload['test']['anomaly_flag_rate']:.2%}")
    for label, stats in eval_payload["by_label"].items():
        print(f"  {label}: n={stats['n']}, flagged={stats['flagged_rate']:.2%}, mean_score={stats['mean_score']:.6f}")


def build_reconstruction_scores(
    x: np.ndarray,
    batch_size: int,
    device_request: str,
) -> np.ndarray:
    device = choose_device(device_request)
    total_batches = math.ceil(x.shape[0] / batch_size)
    print(
        f"Building MOMENT reconstruction scores for {x.shape[0]} windows "
        f"({total_batches} batches, batch_size={batch_size}) on {device}",
        flush=True,
    )

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction"},
    )
    model.init()
    model.to(device)
    model.eval()

    scores: list[np.ndarray] = []
    start_time = time.time()
    with torch.no_grad():
        for batch_index, batch_start in enumerate(range(0, x.shape[0], batch_size), start=1):
            batch = torch.tensor(
                x[batch_start : batch_start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            input_mask = torch.ones(
                (batch.shape[0], batch.shape[2]),
                dtype=torch.float32,
                device=device,
            )
            output = model.detect_anomalies(
                x_enc=batch,
                input_mask=input_mask,
                anomaly_criterion="mse",
            )
            anomaly_scores = getattr(output, "anomaly_scores", None)
            if anomaly_scores is None:
                raise RuntimeError("MOMENT output did not expose anomaly_scores")
            batch_scores = torch.mean(anomaly_scores, dim=(1, 2))
            scores.append(batch_scores.detach().cpu().numpy())
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                elapsed = time.time() - start_time
                print(
                    f"  scored batch {batch_index}/{total_batches} "
                    f"({batch_start + batch.shape[0]}/{x.shape[0]} windows, {elapsed:.1f}s)",
                    flush=True,
                )
    return np.concatenate(scores)


def grouped_train_cal_test_split(
    scores: np.ndarray,
    y: np.ndarray,
    files: np.ndarray,
    test_size: float,
    calibration_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=11)
    train_cal_idx, test_idx = next(splitter.split(scores.reshape(-1, 1), y, groups=files))
    cal_splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=13)
    train_rel_idx, cal_rel_idx = next(
        cal_splitter.split(
            scores[train_cal_idx].reshape(-1, 1),
            y[train_cal_idx],
            groups=files[train_cal_idx],
        )
    )
    train_idx = train_cal_idx[train_rel_idx]
    cal_idx = train_cal_idx[cal_rel_idx]
    return train_idx, cal_idx, test_idx


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
            "reconstruction_error": float(score),
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


def write_predictions(path: Path, predictions: list[dict]) -> None:
    fields = ["label", "y_true", "file", "window_start_s", "reconstruction_error", "flagged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
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
    parser = argparse.ArgumentParser(
        description="Train a lightweight anomaly head on top of pretrained MOMENT embeddings."
    )
    parser.add_argument("--windows", type=Path, default=Path("data/px4_flight_review/moment_windows/px4_moment_windows.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/px4_flight_review/moment_windows/moment_binary_head"))
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head", choices=["logistic", "random_forest"], default="random_forest")
    args = parser.parse_args()

    data = np.load(args.windows, allow_pickle=True)
    x = data["x"]
    y_multiclass = data["y"]
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]
    y = (y_multiclass != 0).astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output_dir / "moment_embeddings.npz"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path, allow_pickle=True)["embeddings"]
    else:
        embeddings = build_moment_embeddings(x, batch_size=args.batch_size)
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings.astype(np.float32),
            y=y,
            labels=labels,
            files=files,
            window_start_s=starts,
        )

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

    clf = build_head(args.head)
    clf.fit(embeddings[train_idx], y[train_idx])

    cal_scores = clf.predict_proba(embeddings[cal_idx])[:, 1]
    test_scores = clf.predict_proba(embeddings[test_idx])[:, 1]
    threshold = normal_quantile_threshold(cal_scores, y[cal_idx], args.threshold_quantile)

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
    eval_payload["interpretation"] = "anomaly_probability > threshold is anomalous"

    joblib.dump(clf, args.output_dir / "moment_binary_head.joblib")
    (args.output_dir / "evaluation.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    write_predictions(args.output_dir / "predictions.csv", eval_payload["predictions"])

    print(f"Wrote model to {args.output_dir / 'moment_binary_head.joblib'}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Test ROC-AUC: {eval_payload['test']['roc_auc']:.4f}")
    print(f"Test accuracy: {eval_payload['test']['accuracy']:.4f}")
    for label, stats in eval_payload["by_label"].items():
        print(f"  {label}: n={stats['n']}, flagged={stats['flagged_rate']:.2%}, mean_score={stats['mean_score']:.4f}")


def build_head(name: str):
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
                ),
            ),
        ]
    )


def build_moment_embeddings(x: np.ndarray, batch_size: int) -> np.ndarray:
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch_start in range(0, x.shape[0], batch_size):
            batch = torch.tensor(x[batch_start : batch_start + batch_size], dtype=torch.float32)
            output = model(x_enc=batch)
            embedding = getattr(output, "embeddings", None)
            if embedding is None:
                raise RuntimeError("MOMENT output did not expose embeddings")
            embeddings.append(embedding.detach().cpu().numpy())
    return np.vstack(embeddings)


def normal_quantile_threshold(scores: np.ndarray, y: np.ndarray, quantile: float) -> float:
    normal_scores = scores[y == 0]
    if normal_scores.size == 0:
        return float(np.quantile(scores, quantile))
    return float(np.quantile(normal_scores, quantile))


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

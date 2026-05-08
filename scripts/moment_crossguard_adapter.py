from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype adapter for feeding PX4 windows into MOMENT or a local fallback anomaly model."
    )
    parser.add_argument("--windows", type=Path, default=Path("data/px4_flight_review/moment_windows/px4_moment_windows.npz"))
    parser.add_argument("--backend", choices=["auto", "moment", "isolation_forest"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("data/px4_flight_review/moment_windows/anomaly_scores.csv"))
    args = parser.parse_args()

    data = np.load(args.windows, allow_pickle=True)
    x = data["x"]
    y = data["y"]
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]

    backend = choose_backend(args.backend)
    if backend == "moment":
        scores = score_with_moment(x)
    else:
        scores = score_with_isolation_forest(x, y)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("label,y,file,window_start_s,anomaly_score,backend\n")
        for label, numeric, file, start, score in zip(labels, y, files, starts, scores):
            handle.write(f"{label},{int(numeric)},{file},{float(start):.3f},{float(score):.6f},{backend}\n")

    print(f"Wrote anomaly scores to {args.output}")
    summarize(labels, scores)
    if backend == "isolation_forest":
        print("Used local IsolationForest fallback. MOMENT hooks are ready, but momentfm/transformers are not installed.")


def choose_backend(requested: str) -> str:
    if requested == "isolation_forest":
        return "isolation_forest"
    if requested == "moment":
        require_moment()
        return "moment"
    try:
        require_moment()
        return "moment"
    except ModuleNotFoundError:
        return "isolation_forest"


def require_moment() -> None:
    import momentfm  # noqa: F401
    import torch  # noqa: F401


def score_with_moment(x: np.ndarray) -> np.ndarray:
    """Use MOMENT embeddings as anomaly features.

    This function is intentionally small and isolated because MOMENT's package
    versions can vary. Once `momentfm` is installed, this is where we adapt the
    online pretrained model instead of hand-building a neural net from scratch.
    """

    import torch
    from momentfm import MOMENTPipeline
    from sklearn.ensemble import IsolationForest

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch_start in range(0, x.shape[0], 16):
            batch = torch.tensor(x[batch_start : batch_start + 16], dtype=torch.float32)
            output = model(x_enc=batch)
            embedding = getattr(output, "embeddings", None)
            if embedding is None:
                embedding = getattr(output, "last_hidden_state", None)
            if embedding is None:
                raise RuntimeError("MOMENT output did not expose embeddings/last_hidden_state")
            embeddings.append(embedding.detach().cpu().numpy().reshape(batch.shape[0], -1))

    features = np.vstack(embeddings)
    detector = IsolationForest(n_estimators=200, contamination="auto", random_state=7)
    detector.fit(features)
    return -detector.score_samples(features)


def score_with_isolation_forest(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    flattened = x.reshape(x.shape[0], -1)
    normal = flattened[y == 0]
    train = normal if normal.size else flattened
    detector = IsolationForest(n_estimators=200, contamination="auto", random_state=7)
    detector.fit(train)
    return -detector.score_samples(flattened)


def summarize(labels: np.ndarray, scores: np.ndarray) -> None:
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        print(
            f"  {label}: n={int(mask.sum())}, "
            f"mean_score={float(np.mean(scores[mask])):.4f}, "
            f"p95={float(np.percentile(scores[mask], 95)):.4f}"
        )


if __name__ == "__main__":
    main()

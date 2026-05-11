from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np

from crossguard.defense.state import DroneState
from crossguard.ml.base import MLSanityDecision
from crossguard.ml.state_features import WindowNormalization, state_window_to_array


Backend = Literal["moment_head", "window_head"]


class MomentStateSanityChecker:
    """Runtime learned sanity checker over rolling UAV state windows.

    The expected production path is:

    1. Build UAV-SEAD windows with the same feature names.
    2. Train MOMENT embeddings plus a lightweight sklearn head.
    3. Load the normalization stats, head, and threshold here.

    A ``window_head`` backend is also supported for lightweight tests or ablations
    where a sklearn model consumes the flattened normalized window directly.
    """

    def __init__(
        self,
        normalization_path: str | Path,
        head_path: str | Path,
        threshold_path: str | Path,
        *,
        seq_len: int = 64,
        backend: Backend = "moment_head",
        moment_model_name: str = "AutonLab/MOMENT-1-large",
        batch_size: int = 1,
    ) -> None:
        self.normalization = WindowNormalization.from_json(normalization_path)
        self.head_path = Path(head_path)
        self.threshold = _load_threshold(threshold_path)
        self.seq_len = seq_len
        self.backend = backend
        self.moment_model_name = moment_model_name
        self.batch_size = batch_size
        self.history: deque[DroneState] = deque(maxlen=seq_len)
        self._head = None
        self._moment_model = None
        self._load_error: str | None = None

    def observe(self, state: DroneState) -> MLSanityDecision:
        self.history.append(state)
        if len(self.history) < self.seq_len:
            return MLSanityDecision(
                ready=False,
                backend=self.backend,
                reason=f"warming up learned checker ({len(self.history)}/{self.seq_len} states)",
            )

        raw_window = state_window_to_array(self.history, self.normalization.feature_names)
        normalized_window = self.normalization.normalize(raw_window)
        score = self._score(normalized_window)
        if score is None:
            return MLSanityDecision(
                ready=False,
                backend=self.backend,
                threshold=self.threshold,
                reason=self._load_error or "learned checker unavailable",
            )

        return MLSanityDecision(
            ready=True,
            alert=score > self.threshold,
            score=score,
            threshold=self.threshold,
            backend=self.backend,
            reason="score > threshold is anomalous",
        )

    def reset(self) -> None:
        self.history.clear()

    def _score(self, normalized_window: np.ndarray) -> float | None:
        head = self._load_head()
        if head is None:
            return None

        if self.backend == "window_head":
            features = normalized_window.reshape(1, -1)
        else:
            embedding = self._moment_embedding(normalized_window)
            if embedding is None:
                return None
            features = embedding.reshape(1, -1)

        if hasattr(head, "predict_proba"):
            proba = head.predict_proba(features)
            return float(proba[0, -1])
        if hasattr(head, "decision_function"):
            return float(head.decision_function(features)[0])
        if hasattr(head, "score_samples"):
            return float(-head.score_samples(features)[0])

        self._load_error = "loaded ML head does not expose predict_proba, decision_function, or score_samples"
        return None

    def _load_head(self):
        if self._head is not None:
            return self._head
        if not self.head_path.exists():
            self._load_error = f"missing ML head artifact: {self.head_path}"
            return None
        try:
            import joblib
        except ModuleNotFoundError as exc:
            self._load_error = f"joblib is not installed: {exc}"
            return None
        self._head = joblib.load(self.head_path)
        return self._head

    def _moment_embedding(self, normalized_window: np.ndarray) -> np.ndarray | None:
        try:
            import torch
            from momentfm import MOMENTPipeline
        except ModuleNotFoundError as exc:
            self._load_error = f"MOMENT runtime dependencies are not installed: {exc}"
            return None

        if self._moment_model is None:
            model = MOMENTPipeline.from_pretrained(
                self.moment_model_name,
                model_kwargs={"task_name": "embedding"},
            )
            model.init()
            model.eval()
            self._moment_model = model

        with torch.no_grad():
            batch = torch.tensor(normalized_window[None, :, :], dtype=torch.float32)
            output = self._moment_model(x_enc=batch)
            embedding = getattr(output, "embeddings", None)
            if embedding is None:
                embedding = getattr(output, "last_hidden_state", None)
            if embedding is None:
                self._load_error = "MOMENT output did not expose embeddings or last_hidden_state"
                return None
            return embedding.detach().cpu().numpy()


def _load_threshold(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "threshold" not in payload:
        raise ValueError(f"{path} must contain a top-level 'threshold'")
    return float(payload["threshold"])

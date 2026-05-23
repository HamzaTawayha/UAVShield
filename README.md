# UAVShield

UAVShield is a learned sanity-checker for UAV agent memory and telemetry. The
current detector is meant to sit between an attack injector and the planner:

```text
telemetry / perception / memory update -> UAVShield detector -> allow or quarantine -> planner
```

## Current Defense Artifact

The current best UAV-SEAD detector is:

```text
Statistical UAV telemetry features + HistGradientBoosting
+ per-anomaly-family thresholds searched over 1000 calibration configs
```

Held-out UAV-SEAD result:

```text
Accuracy:          93.76%
Precision:         88.57%
Recall:            71.26%
F1:                0.7898
ROC-AUC:           0.9591
Normal flag rate:  1.81%
```

Selected anomaly-family thresholds:

```text
External Position: 0.05
Global Position:   0.75
Altitude:          0.05
Mechanical:        0.60
```

## Rebuild The Detector

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/uav_sead_build_moment_windows.py \
  --include-log-level-anomalies \
  --output data/uav_sead/moment_windows/uav_sead_precise_windows.npz

python scripts/augment_uav_sead_physics_features.py

python scripts/search_family_threshold_configs.py \
  --windows data/uav_sead/moment_windows/uav_sead_precise_physics_windows.npz \
  --output-dir data/uav_sead/moment_windows/statistical_features_family_threshold_search_1000 \
  --old-eval data/uav_sead/moment_windows/statistical_features_histgb/evaluation.json \
  --model hist_gb \
  --threshold-step 0.05 \
  --trials 1000 \
  --selection-metric accuracy
```

The main local artifact is:

```text
data/uav_sead/moment_windows/statistical_features_family_threshold_search_1000/family_threshold_search_model.joblib
```

Raw UAV-SEAD logs and generated model artifacts are intentionally ignored by
Git because they are large and machine-local.

## For Attack Testing

If you are testing attacks against this defense, start here:

[PhD attack integration guide](docs/PHD_ATTACK_INTEGRATION.md)

The short version:

1. Keep your attack runner unchanged until it is about to expose a telemetry,
   perception, or memory update to the planner.
2. Build/update a rolling UAV telemetry window.
3. Call the UAVShield detector.
4. If the detector flags the update, quarantine it and do not write it into the
   planner memory.
5. Report defense detection rate, false positive rate, false negative rate, and
   mission recovery compared with the original attack-only baseline.

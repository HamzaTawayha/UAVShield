# MOMENT Adaptation Pipeline

This folder now contains the first step toward adapting an online pretrained
time-series model, MOMENT, to CrossGuard telemetry.

## Step 1: Build Time-Series Windows

Convert PX4 `.ulg` files into fixed-length multivariate windows:

```powershell
python .\scripts\px4_build_moment_windows.py `
  --limit-per-class 60 `
  --seq-len 64 `
  --stride 32 `
  --max-windows-per-log 15
```

Output:

```text
data/px4_flight_review/moment_windows/px4_moment_windows.npz
```

Current build:

```text
x = (675, 12, 64)
normal_good = 240 windows
unsatisfactory = 267 windows
crash = 168 windows
```

Shape convention:

```text
x = (windows, channels, timesteps)
```

Current channels:

```text
local_x_m
local_y_m
local_z_m
local_vx_mps
local_vy_mps
local_vz_mps
accel_x_mps2
accel_y_mps2
accel_z_mps2
battery_remaining_pct
battery_current_a
battery_voltage_v
```

Labels:

```text
0 = normal_good
1 = unsatisfactory
2 = crash
```

## Step 2: Score Windows

Local fallback:

```powershell
python .\scripts\moment_crossguard_adapter.py --backend isolation_forest
```

MOMENT backend, once dependencies are installed:

```powershell
python -m pip install "transformers<5" huggingface-hub einops tqdm torch
python -m pip install momentfm --no-deps
python .\scripts\moment_crossguard_adapter.py --backend moment `
  --output .\data\px4_flight_review\moment_windows\moment_anomaly_scores.csv
```

The MOMENT adapter uses the pretrained online model `AutonLab/MOMENT-1-large`
as an embedding model and trains a small anomaly detector on top of its learned
representations. This satisfies the professor's direction: we are adapting an
online model instead of building a neural network from scratch.

Current MOMENT embedding + unsupervised detector result:

```text
threshold = 0.464598 calibrated from normal_good q=0.95
normal_good false-positive estimate = 5.00%
unsatisfactory flagged rate = 20.97%
crash flagged rate = 16.67%
```

## Step 3: Calibrate a Threshold

Raw anomaly scores are not enough. We need a deployable threshold.

Calibrate from the normal-flight windows:

```powershell
python .\scripts\calibrate_px4_anomaly_threshold.py --quantile 0.95
```

Output:

```text
data/px4_flight_review/moment_windows/anomaly_threshold.json
data/px4_flight_review/moment_windows/anomaly_threshold.evaluation.csv
```

Interpretation:

```text
score > threshold => ML anomaly violation
```

The default `0.95` threshold means: choose the 95th percentile of the
`normal_good` scores. In other words, tolerate about 5% false positives on the
normal calibration set, then measure how many unsatisfactory/crash windows are
flagged.

Current fallback result:

```text
threshold = 0.612010
normal_good false-positive estimate = 5.71%
unsatisfactory flagged rate = 0.00%
crash flagged rate = 1.54%
```

This fallback result is weak. That is useful to know: the basic IsolationForest
over raw flattened windows is only a pipeline sanity check, not the final model.
The MOMENT embedding result above is stronger and is the result to show as the
first real pretrained-model adaptation.

## Step 4: Train a Lightweight MOMENT Head

This adapts the online pretrained model without building a neural network from
scratch. MOMENT produces embeddings, then a small sklearn head learns normal vs
abnormal PX4 windows.

```powershell
python .\scripts\train_moment_binary_head.py
```

Outputs:

```text
data/px4_flight_review/moment_windows/moment_binary_head/moment_embeddings.npz
data/px4_flight_review/moment_windows/moment_binary_head/moment_binary_head.joblib
data/px4_flight_review/moment_windows/moment_binary_head/evaluation.json
data/px4_flight_review/moment_windows/moment_binary_head/predictions.csv
```

Interpretation:

```text
anomaly_probability > threshold => ML anomaly violation
```

Current best result, using `AutonLab/MOMENT-1-large` embeddings with a
random-forest head:

```text
threshold = 0.864022 calibrated from normal_good q=0.95
test ROC-AUC = 0.7745
test accuracy = 0.6604
normal_good false-positive estimate = 2.78%
unsatisfactory flagged rate = 51.61%
crash flagged rate = 7.14%
```

The random-forest head is currently better than the logistic-regression head for
these logs. It is strong on `unsatisfactory` windows and conservative on normal
windows. The crash rate is still low because many windows from a crash-labeled
log are normal-looking pre-crash flight; the next improvement is to make
event-centered crash windows instead of labeling every crash-log window equally.

## Why This Matters

CrossGuard rules remain the hard safety guardrail. MOMENT learns the normal
telemetry envelope from real PX4 windows and contributes adaptive anomaly scores
for patterns that fixed rules might miss.

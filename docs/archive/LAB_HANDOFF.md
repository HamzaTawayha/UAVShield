# UAVShield Lab Handoff

## Current Status

This repository contains the CrossGuard/UAVShield prototype:

- Rule-based UAV sanity checker for memory-poisoned telemetry.
- Streamlit demo UI for testing injected vs previous drone states.
- PX4 Flight Review data pipeline.
- MOMENT-based time-series anomaly pipeline.
- UAV-SEAD wide-state window builder for dynamic ML sanity checking.
- Small saved MOMENT-derived artifacts from the local PX4 experiment.

Raw `.ulg` datasets are intentionally not committed because they are too large
for GitHub. Download them again on the lab server.

## What Was Built

### Rule-Based Harness

Main files:

- `crossguard/defense/invariants.py`
- `crossguard/defense/harness.py`
- `crossguard/defense/state.py`
- `crossguard/defense/scoring.py`

The harness checks common-sense physical consistency, including:

- impossible GPS jumps
- impossible battery drops/rises
- altitude-rate violations
- velocity vs position mismatch
- acceleration spikes
- IMU vs velocity mismatch
- heading vs velocity mismatch
- stale/future/replayed sensor timestamps
- geofence violations
- command plausibility
- ground-altitude sanity
- multi-drone consistency
- perception/depth mismatch

### UI Demo

Run:

```bash
streamlit run streamlit_app.py
```

### MOMENT Pipeline

Main files:

- `scripts/uav_sead_build_moment_windows.py`
- `scripts/px4_build_moment_windows.py`
- `scripts/moment_crossguard_adapter.py`
- `scripts/calibrate_px4_anomaly_threshold.py`
- `scripts/train_moment_binary_head.py`
- `data/px4_flight_review/MOMENT_PIPELINE.md`

The UAV-SEAD builder is now the preferred path for the professor's requested
dynamic sanity checker. It extracts broad state windows from real UAV logs:
position, velocity, acceleration, battery, distance/range sensor, barometer,
gyroscope, magnetometer, estimator health, actuator outputs, mission state,
setpoints, and telemetry-link health.

Current local result:

```text
MOMENT embeddings + RandomForest head
ROC-AUC: 0.7745
Normal false positives: 2.78%
Unsatisfactory flagged: 51.61%
Crash flagged: 7.14%
Threshold: 0.864022
```

Simple interpretation:

- `ROC-AUC 0.7745`: decent separation between normal and abnormal flight windows.
- `Normal false positives 2.78%`: low false alarms on normal windows.
- `Unsatisfactory flagged 51.61%`: catches about half of unsatisfactory windows.
- `Crash flagged 7.14%`: weak crash detection for now.
- `Threshold 0.864022`: above this anomaly probability, flag the window.

Crash detection is weak because a crash-labeled log contains many normal-looking
pre-crash windows. The next improvement is event-centered crash windowing.

## Server Setup

Clone:

```bash
git clone https://github.com/HamzaTawayha/UAVShield.git
cd UAVShield
```

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install momentfm --no-deps
```

If the server uses Conda:

```bash
conda create -n uavshield python=3.12 -y
conda activate uavshield
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install momentfm --no-deps
```

## Download Restricted UAV Dataset

After Hugging Face access is approved, authenticate on the lab server:

```bash
huggingface-cli login
```

Then download:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aykutkabaoglu/uav-flight-anomaly-dataset",
    repo_type="dataset",
    local_dir="data/uav_sead",
    local_dir_use_symlinks=False,
)
PY
```

The `.gitignore` excludes `data/uav_sead/ulg_files/`, so the downloaded logs
will stay local to the server.

## Rebuild PX4 MOMENT Experiment

If PX4 logs are present:

```bash
python scripts/px4_build_moment_windows.py --limit-per-class 60 --seq-len 64 --stride 32 --max-windows-per-log 15
python scripts/moment_crossguard_adapter.py --backend moment --output data/px4_flight_review/moment_windows/moment_anomaly_scores.csv
python scripts/calibrate_px4_anomaly_threshold.py --scores data/px4_flight_review/moment_windows/moment_anomaly_scores.csv --output data/px4_flight_review/moment_windows/moment_anomaly_threshold.json --quantile 0.95
python scripts/train_moment_binary_head.py
```

## Build UAV-SEAD Dynamic State Windows

After downloading UAV-SEAD into `data/uav_sead`:

```bash
python scripts/uav_sead_build_moment_windows.py --include-log-level-anomalies
python scripts/train_moment_binary_head.py \
  --windows data/uav_sead/moment_windows/uav_sead_state_windows.npz \
  --output-dir data/uav_sead/moment_windows/moment_binary_head
```

Runtime integration uses `MomentStateSanityChecker` and adds an
`ml.state_anomaly` violation when the learned whole-state model flags the
rolling state window.

## What To Tell Ibrahim

The ROS2 side mainly needs adapters:

1. Subscribe to real drone telemetry topics.
2. Convert telemetry into `DroneState` for the rule-based harness.
3. Maintain a rolling 12-channel time-series buffer for the MOMENT pipeline.
4. Normalize the buffer using `px4_moment_windows.normalization.json`.
5. Run MOMENT embeddings plus the saved anomaly head.
6. Emit a CrossGuard violation if either the deterministic rules or ML score
   indicate an impossible/suspicious state.

Recommended framing:

```text
CrossGuard combines deterministic physics/safety invariants with a pretrained
MOMENT time-series model adapted to UAV telemetry. The rules catch explicit
impossibilities, while MOMENT catches learned deviations from normal flight
patterns.
```

## Immediate Next Steps

1. Download the newly approved restricted dataset on the lab server.
2. Run the UAV-SEAD wide-state converter and inspect the class/window counts.
3. Retrain the MOMENT head using the UAV-SEAD labels and precise anomaly ranges.
4. Compare learned dynamic detection against the fixed rule-only harness.
5. Run final evaluation tables for the paper.

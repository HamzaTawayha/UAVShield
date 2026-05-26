# UAVShield Agent-Log Detector Package

This is the minimal package for Ibrahim to train/evaluate Hamzeh's S1-S4 detector on UAV planner attack logs.

## Required Input

Use the full logs, not only `metrics.csv`.

```text
metrics.csv
results_parent/
  s1/.../<run_id>/events.jsonl
  s2/.../<run_id>/events.jsonl
  s3/.../<run_id>/events.jsonl
  s4/.../<run_id>/events.jsonl
```

If the scenario folders are split across sibling folders, pass their common parent as `--results-dir`. The script searches recursively.

Labels come from `metrics.csv`:

```text
clean      -> 0
s*_attack  -> 1
```

Features come mainly from `events.jsonl`: tool calls, memory writes, observation differences, telemetry summaries, safety-status changes, repair behavior, and planner rationale text.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Recommended Run

```bash
./run_detector.sh /path/to/results_parent /path/to/metrics.csv detector_output
```

Equivalent explicit command:

```bash
python train_agent_log_detector.py \
  --results-dir /path/to/results_parent \
  --metrics-csv /path/to/metrics.csv \
  --require-events \
  --supervised-model extra_trees \
  --threshold-metric accuracy \
  --output-dir detector_output
```

`--require-events` is important: it fails if a `metrics.csv` row has no matching `events.jsonl`.

Expected sanity check:

```text
matched events.jsonl for 1920
metrics-only rows for 0
```

## Current Split

Default split is seed-heldout:

```text
Train seeds:      42,123,256,512,1024,2048
Validation seeds: 4096,8192
Test seeds:       111,222
```

This is preferred for reporting because test seeds are unseen during training and threshold calibration.

## Current Expected Result

With the full S1-S4 logs:

```text
Accuracy:   90.10%
Precision:  85.98%
Recall:     95.83%
F1:         90.64%
ROC-AUC:    97.45%
PR-AUC:     97.54%
```

## Outputs

```text
detector_output/detection_metrics.csv
detector_output/supervised_evaluation.json
detector_output/supervised_predictions.csv
detector_output/supervised_top_features.csv
detector_output/agent_log_detectors.joblib
detector_output/*.png and *.pdf plots
```

## Leakage Policy

The detector excludes oracle fields:

```text
attack
injected_payload
false_belief_label
unsafe_tool_label
termination_reason
```

From `metrics.csv`, it avoids post-hoc attack/evaluation columns such as `fbar`, `rmfr`, `uter`, ghost counters, AIM-MCM metrics, and ground-truth person-distance fields.

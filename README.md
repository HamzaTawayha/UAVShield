# UAVShield

UAVShield is Hamzeh's ML detector for Ibrahim's UAV planner attack logs. The current repo focus is the **supervised S1-S4 agent-log detector** trained on Ibrahim's `metrics.csv` plus per-run `events.jsonl` files.

The detector answers:

```text
Given one completed UAV planner run, was it clean or attacked?
```

It currently predicts:

```text
0 = clean / non-malicious
1 = attacked / malicious
```

It does not currently predict the attack family as a separate model output. The S1-S4 labels are used for per-scenario evaluation breakdowns.

## Current Best Result

Best supervised detector:

```text
Feature extractor: agent-log/runtime features from events.jsonl + safe metrics.csv columns
Model:             ExtraTreesClassifier
Split:             seed-heldout
Threshold:         selected on validation accuracy
```

Held-out test result:

```text
Accuracy:   90.10%
Precision:  85.98%
Recall:     95.83%
F1:         90.64%
ROC-AUC:    97.45%
PR-AUC:     97.54%
```

Per-attack-family detection rate on the held-out test split:

```text
S1:  97.92%
S2: 100.00%
S3: 100.00%
S4:  85.42%
```

## What Data Is Required

The detector needs both:

```text
metrics.csv      -> run list, labels, safe aggregate fields
events.jsonl     -> real per-step detector evidence
```

Expected full-log layout:

```text
results_parent/
  metrics.csv
  s1/.../<run_id>/events.jsonl
  s2/.../<run_id>/events.jsonl
  s3/.../<run_id>/events.jsonl
  s4/.../<run_id>/events.jsonl
```

The local dataset is currently split across sibling folders, so the command uses the parent directory as `--results-dir`:

```text
../S1 and S2(2)/s1
../S1 and S2(2)/s2
../S3 and S4(1)/s3
../S3 and S4(1)/s4
```

The script verifies this with:

```text
matched events.jsonl for 1920
metrics-only rows for 0
```

## Run The Best Detector

From the repo root:

```bash
cd /home/tawayha/Desktop/UAVShield/UAVShield
source .venv/bin/activate

python scripts/train_agent_log_detector.py \
  --results-dir .. \
  --metrics-csv ../metrics.csv \
  --require-events \
  --supervised-model extra_trees \
  --threshold-metric accuracy \
  --output-dir reports/agent_log_detector_supervised_only_extra_trees_accuracy
```

Key outputs:

```text
reports/agent_log_detector_supervised_only_extra_trees_accuracy/detection_metrics.csv
reports/agent_log_detector_supervised_only_extra_trees_accuracy/supervised_evaluation.json
reports/agent_log_detector_supervised_only_extra_trees_accuracy/supervised_predictions.csv
reports/agent_log_detector_supervised_only_extra_trees_accuracy/supervised_top_features.csv
reports/agent_log_detector_supervised_only_extra_trees_accuracy/*.png
reports/agent_log_detector_supervised_only_extra_trees_accuracy/agent_log_detectors.joblib
```

## Train/Test Split

Default split is seed-heldout:

```text
Train seeds:      42, 123, 256, 512, 1024, 2048
Validation seeds: 4096, 8192
Test seeds:       111, 222
```

With the full S1-S4 dataset:

```text
Train:      1152 runs = 576 clean + 576 attack
Validation: 384 runs  = 192 clean + 192 attack
Test:       384 runs  = 192 clean + 192 attack
Total:      1920 runs
```

This is the preferred split for reporting because the test seeds are never used during training or threshold calibration.

## What Features Are Used

The supervised detector extracts run-level features from the logs, including:

```text
tool-call counts
memory-write counts
memory types and keys
planner repair/validation behavior
safety-status changes
true-vs-agent-visible observation differences
detection-count changes
telemetry summaries
LLM rationale text as TF-IDF features
safe aggregate runtime columns from metrics.csv
```

The detector explicitly excludes oracle/label leakage fields:

```text
attack
injected_payload
false_belief_label
unsafe_tool_label
termination_reason
```

For `metrics.csv`, it also avoids post-hoc attack/evaluation fields such as `fbar`, `rmfr`, `uter`, ghost counters, AIM-MCM defense metrics, and ground-truth person-distance columns.

## Package For Ibrahim

A small sendable package is included here:

```text
packages/agent_log_detector_for_ibrahim/
packages/uavshield_agent_log_detector_for_ibrahim.zip
```

It contains only:

```text
train_agent_log_detector.py
README.md
requirements.txt
run_detector.sh
```

See:

[packages/agent_log_detector_for_ibrahim/README.md](packages/agent_log_detector_for_ibrahim/README.md)

Ibrahim's integration guide:

[docs/IBRAHIM_ATTACK_INTEGRATION.md](docs/IBRAHIM_ATTACK_INTEGRATION.md)

## Older Telemetry Work

The repository still contains earlier UAV-SEAD / MOMENT / telemetry anomaly detection scripts. Those are not the current Ibrahim handoff path. The current paper-facing detector is:

```text
scripts/train_agent_log_detector.py
```

The older telemetry detector can remain as background work, but it should not be presented to Ibrahim as the integration target for the current S1-S4 attack results.

# UAVShield

This repository is organized as Hamzeh's handoff to Ibrahim for the UAV planner attack detector.

## Start Here

Use this folder:

[ibrahim_handoff](ibrahim_handoff)

It contains the minimal runnable detector package:

```text
ibrahim_handoff/
  README.md
  train_agent_log_detector.py
  run_detector.sh
  requirements.txt
```

## What The Detector Does

The current detector is a supervised S1-S4 agent-log detector trained on Ibrahim's attack-run outputs:

```text
Input:  metrics.csv + per-run events.jsonl files
Output: clean vs attacked
Model:  ExtraTreesClassifier
```

Prediction target:

```text
0 = clean / non-malicious
1 = attacked / malicious
```

The model is binary. It does not separately predict S1/S2/S3/S4, but the evaluation reports detection rate for each attack family.

## Current Best Result

Seed-heldout test split:

```text
Accuracy:   90.10%
Precision:  85.98%
Recall:     95.83%
F1:         90.64%
ROC-AUC:    97.45%
PR-AUC:     97.54%
```

Per-family detection:

```text
S1:  97.92%
S2: 100.00%
S3: 100.00%
S4:  85.42%
```

## Run Command

From the repo root:

```bash
source .venv/bin/activate

python scripts/train_agent_log_detector.py \
  --results-dir /path/to/results_parent \
  --metrics-csv /path/to/metrics.csv \
  --require-events \
  --supervised-model extra_trees \
  --threshold-metric accuracy \
  --output-dir reports/agent_log_detector_supervised_only_extra_trees_accuracy
```

Expected data sanity check:

```text
matched events.jsonl for 1920
metrics-only rows for 0
```

## Repository Map

```text
ibrahim_handoff/                  clean package to send/use
scripts/train_agent_log_detector.py  source detector implementation
docs/IBRAHIM_ATTACK_INTEGRATION.md   detailed integration notes
docs/archive/                     old planning docs and paper drafts
crossguard/                       older runtime/telemetry scaffolding
data/                             older telemetry datasets/artifacts
scripts/                          detector script plus older experiments
```

For Ibrahim, the only folder he should need is:

```text
ibrahim_handoff/
```

## Notes On Data

The detector requires the full per-run logs, not just `metrics.csv`.

Required:

```text
metrics.csv
events.jsonl for every row in metrics.csv
```

`--require-events` is used so the run fails if any event log is missing.

## Legacy Work

Older UAV-SEAD, MOMENT, telemetry, Darts, and iTransformer experiments are still in the repo for reference. They are not the current Ibrahim handoff path.

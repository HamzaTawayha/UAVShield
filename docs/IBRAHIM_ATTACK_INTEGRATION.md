# Ibrahim Attack Integration Guide

This guide is for Ibrahim's attack runner/results. It explains exactly what Hamzeh's detector needs, how to run it, and what outputs to send back.

## Current Detector

Current detector:

```text
Supervised ExtraTrees agent-log detector
```

Prediction target:

```text
clean      -> 0
s*_attack  -> 1
```

The model is binary. It flags malicious vs non-malicious runs. S1/S2/S3/S4 are used for reporting per-attack-family detection rates.

## Files Needed From The Attack Runner

For every run listed in `metrics.csv`, include:

```text
events.jsonl
summary.json
trajectory.csv
```

Minimum required file:

```text
events.jsonl
```

`metrics.csv` alone is not enough for the best detector because it is only a run-level scoreboard. The `events.jsonl` files contain the real evidence: observations, memory writes, tool calls, telemetry, planner decisions, repairs, and rationale text.

Expected shape:

```text
results_parent/
  metrics.csv
  s1/<model>/planning/<run_id>/events.jsonl
  s1/<model>/planning/<run_id>/summary.json
  s1/<model>/planning/<run_id>/trajectory.csv
  s2/<model>/planning/<run_id>/events.jsonl
  s3/<model>/planning/<run_id>/events.jsonl
  s4/<model>/planning/<run_id>/events.jsonl
```

If scenarios are split across sibling folders, pass the parent folder as `--results-dir`. The script searches recursively.

## Run Command

From the UAVShield repo:

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

`--require-events` is intentional. It makes the script fail if any row in `metrics.csv` does not have a matching `events.jsonl`, preventing accidental weak CSV-only evaluation.

Expected sanity check:

```text
Loaded 1920 rows from ../metrics.csv
matched events.jsonl for 1920
metrics-only rows for 0
```

## Current Result

Seed-heldout test result:

```text
Accuracy:   90.10%
Precision:  85.98%
Recall:     95.83%
F1:         90.64%
ROC-AUC:    97.45%
PR-AUC:     97.54%
```

Per-family held-out detection:

```text
S1:  97.92%
S2: 100.00%
S3: 100.00%
S4:  85.42%
```

## Outputs

The detector writes:

```text
detection_metrics.csv
supervised_evaluation.json
supervised_predictions.csv
supervised_top_features.csv
agent_log_detectors.joblib
agent-log-detector-metrics.png/pdf
supervised-confusion-matrix.png/pdf
supervised-roc.png/pdf
supervised-pr.png/pdf
supervised-flagged-rate-by-family.png/pdf
```

Use `detection_metrics.csv` for tables and the PNG/PDF files for slides.

## What The Detector Uses

It extracts features from `events.jsonl`, including:

```text
tool calls
memory writes
memory type/key counts
planner repair behavior
parse/validation behavior
safety status changes
true-vs-agent-visible observation differences
detection count differences
telemetry summaries
LLM rationale text using TF-IDF
```

It also uses safe aggregate runtime columns from `metrics.csv`, such as:

```text
steps
elapsed_s
total_distance_m
memory_citation_count
total_llm_calls
parse_success_rate
LLM latency statistics
planner_type
memory_mode
current_safety_status
```

## Leakage Policy

The detector does not use oracle fields as features:

```text
attack
injected_payload
false_belief_label
unsafe_tool_label
termination_reason
```

From `metrics.csv`, it avoids post-hoc attack/evaluation fields:

```text
uter_*
fbar_*
rmfr
executed_ghost_goto_count
ghost_per_step
raw_llm_ghost_goto_intent_count
ddr/fpr/fnr
aim_mcm_checks_*
distance-to-true-person fields
mission_success / mission_failure
```

## How To Interpret The Output

Each row in `supervised_predictions.csv` contains:

```text
run_id
scenario
condition
memory_mode
y_true      # 0 clean, 1 attack
score       # model attack score/probability
threshold   # selected threshold
flagged     # 0 allowed, 1 malicious
events_path
```

For the current defense story:

```text
flagged = 1 -> detector would quarantine/escalate this run/update
flagged = 0 -> detector treats it as clean
```

This is currently an offline run-level detector over completed logs. To convert it into a live defense, Ibrahim's runner should call the same feature extractor before committing agent-visible observations/memory into the planner state, then quarantine updates whose score crosses the threshold.

## Handoff Package

For a minimal copy that can be sent without the full repo, use:

```text
packages/uavshield_agent_log_detector_for_ibrahim.zip
```

It contains the detector script, a README, requirements, and a runner shell script.

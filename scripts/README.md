# Scripts

Current handoff script:

```text
train_agent_log_detector.py
```

Run it from the repository root with:

```bash
python scripts/train_agent_log_detector.py \
  --results-dir .. \
  --metrics-csv ../metrics.csv \
  --require-events \
  --supervised-model extra_trees \
  --threshold-metric accuracy \
  --output-dir reports/agent_log_detector_supervised_only_extra_trees_accuracy
```

Other tracked scripts in this folder are earlier telemetry, MOMENT, Darts, iTransformer, dataset-building, or reporting experiments.

Older local helper scripts that are not part of the Ibrahim handoff are parked in:

```text
legacy/
```

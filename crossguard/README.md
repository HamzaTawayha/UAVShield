# CrossGuard Runtime Notes

This folder contains older CrossGuard runtime scaffolding and earlier telemetry/MOMENT experiments.

For the current Ibrahim handoff, use the supervised S1-S4 agent-log detector instead:

```text
../scripts/train_agent_log_detector.py
```

Start here:

```text
../README.md
../docs/IBRAHIM_ATTACK_INTEGRATION.md
```

Current best run:

```bash
python scripts/train_agent_log_detector.py \
  --results-dir .. \
  --metrics-csv ../metrics.csv \
  --require-events \
  --supervised-model extra_trees \
  --threshold-metric accuracy \
  --output-dir reports/agent_log_detector_supervised_only_extra_trees_accuracy
```

The older telemetry detector is retained for background/reference only. It is not the current integration target for Ibrahim's S1-S4 attack results.

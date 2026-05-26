#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-.}"
METRICS_CSV="${2:-metrics.csv}"
OUTPUT_DIR="${3:-detector_output}"
PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" train_agent_log_detector.py \
  --results-dir "$RESULTS_DIR" \
  --metrics-csv "$METRICS_CSV" \
  --require-events \
  --supervised-model extra_trees \
  --threshold-metric accuracy \
  --output-dir "$OUTPUT_DIR"

#!/usr/bin/env bash

CONFIG_FILE="${1##*/}"
CONFIG_DIR="${1%"$CONFIG_FILE"}"

echo ""
echo "Running experiment: $CONFIG_FILE"
echo ""

EXPERIMENT_LOG_FOLDER="logs/$(date +%s)/${CONFIG_FILE%.yaml}"

uv run main.py \
  --config "$CONFIG_DIR/$CONFIG_FILE" \
  --resume \
  --log-dir "${EXPERIMENT_LOG_FOLDER}" \
  --verbose

uv run src/tools/generate_graphs.py \
  --input-dir "${EXPERIMENT_LOG_FOLDER}/training/metrics/validation" \
  --output-dir "${EXPERIMENT_LOG_FOLDER}/training/graphs/validation"

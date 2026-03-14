#!/usr/bin/env bash

CONFIG_FILE="${1##*/}"
CONFIG_DIR="${1%"$CONFIG_FILE"}"

echo ""
echo "Running experiment: $CONFIG_FILE"
echo ""

uv run main.py \
  --config "$CONFIG_DIR/$CONFIG_FILE" \
  --resume \
  --log-dir logs/"${CONFIG_FILE%.yaml}"

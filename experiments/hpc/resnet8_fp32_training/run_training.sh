#!/bin/bash
# Usage (in the repo, after `module load git uv`):
#   uv sync
#   sbatch run_training.sh [config.yaml] [seed]
#SBATCH --job-name=resnet8_cifar10_fp32_training_DF
#SBATCH --partition=GPU-small
#SBATCH --time=00:45:00
#SBATCH --mem=6G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=/lustre/scratch/smdafede-resnet8_training/%j/resnet8_cifar10_fp32.out
#SBATCH --error=/lustre/scratch/smdafede-resnet8_training/%j/resnet8_cifar10_fp32.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# sbatch copies this script, so start from the submit directory
cd "$SLURM_SUBMIT_DIR" || exit 1
# resolve a given config path before leaving the submit directory
CONFIG="${1:+$(realpath "$1")}"
module load git uv
cd "$(git rev-parse --show-toplevel)" || exit 1

# config file: first argument (sbatch run_training.sh <config.yaml>), else config.yaml
CONFIG="${CONFIG:-$PWD/experiments/hpc/resnet8_fp32_training/config.yaml}"

# variables that config.yaml refers to
export WS="$(ws_find resnet8_training)"
export RUN_DIR="$WS/$SLURM_JOB_ID"

# seed: second argument, else the entry of the array task, else the config
SEEDS=(42 0 1 2 3 7 123 1337 2024 31415)
SEED="${2:-}"
if [ -z "$SEED" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]:-}"
    if [ -z "$SEED" ]; then
        echo "no seed for array task $SLURM_ARRAY_TASK_ID" >&2
        exit 1
    fi
fi
SEED_ARGS=()
if [ -n "$SEED" ]; then
    SEED_ARGS=(--seed "$SEED")
fi

echo "config: $CONFIG"
echo "seed: ${SEED:-from config}"
echo ""

# the environment is not synced here: the array tasks share it, so run
# `uv sync` once in the repo before sbatch (see the usage in the header)
uv run --no-sync experiments/hpc/resnet8_fp32_training/training.py "$CONFIG" "${SEED_ARGS[@]}"

echo ""
echo "Job finished at: $(date)"

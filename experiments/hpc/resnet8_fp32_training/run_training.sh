#!/bin/bash
#SBATCH --job-name=resnet8_cifar10_fp32_training_DF
#SBATCH --partition=STD-s-96h
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=resnet8_cifar10_fp32_%j.out
#SBATCH --error=resnet8_cifar10_fp32_%j.err

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
export RUN_DIR="$WS/$(date +%s)"

echo "config: $CONFIG"
echo ""

uv sync
echo "environment setup done"
echo ""

uv run experiments/hpc/resnet8_fp32_training/training.py "$CONFIG"

echo ""
echo "Job finished at: $(date)"

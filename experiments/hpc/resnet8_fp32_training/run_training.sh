#!/bin/bash
#SBATCH --job-name=resnet8_cifar10_fp32_training_DF
#SBATCH --partition=STD-s-96h
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=resnet8_cifar10_fp32_%j.out
#SBATCH --error=resnet8_cifar10_fp32_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

WS="$(ws_find resnet8_training)"
WS_PRJ="$WS/$(date +%s)"
export DATASET_SRC="$WS/datasets/cifar10/cifar10.h5"
export LOG_DIR="$WS_PRJ/logs"
export SEED="42"

echo "environment variables set"
echo ""

module load git uv
uv sync
echo "uv synced"
echo ""

uv run training.py

echo ""
echo "Job finished at: $(date)"

#!/bin/bash
#SBATCH --job-name=afrihubert-gusii
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/afrihubert-%j.out
#SBATCH --error=logs/afrihubert-%j.err

set -euo pipefail

cd /home/mila/a/ahomagnr/low-resource-asr-benchmark

source /home/mila/a/ahomagnr/venvs/asr-benchmark-clean/bin/activate

# Hugging Face cache on scratch
export HF_HOME=/network/scratch/a/ahomagnr/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

CONFIG="${1:?Usage: sbatch scripts/train_afrihubert.sbatch CONFIG}"

echo "=================================="
echo "AfriHuBERT - Full ASR Training"
echo "Configuration: $CONFIG"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=================================="

nvidia-smi

python -m src.finetuning.afrihubert_train \
    --config "$CONFIG"

echo "Training finished: $(date)"

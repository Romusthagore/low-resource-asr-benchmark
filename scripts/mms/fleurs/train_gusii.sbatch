#!/bin/bash
#SBATCH --job-name=mms-gusii
#SBATCH --partition=long
#SBATCH --constraint=turing
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mms-gusii-%j.out
#SBATCH --error=logs/mms-gusii-%j.err

set -euo pipefail

cd ~/low-resource-asr-benchmark

source /home/mila/a/ahomagnr/venvs/asr-benchmark-clean/bin/activate

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

export HF_HOME=/network/scratch/a/ahomagnr/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export TMPDIR=/network/scratch/a/ahomagnr/tmp

mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TMPDIR"
mkdir -p logs

OUTPUT="/network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/mms/fleurs/gusii/run_001"
mkdir -p "$OUTPUT"

echo "=== MMS-1B Gusii FLEURS ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

nvidia-smi

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m src.finetuning.mms_train \
    --config configs/mms.yaml \
    --output-dir "$OUTPUT"

echo "Training completed: $(date)"

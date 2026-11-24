#!/bin/bash
#SBATCH --job-name=afrihubert-large-kamba
#SBATCH --partition=long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/afrihubert-large-kamba-%j.out
#SBATCH --error=logs/afrihubert-large-kamba-%j.err

set -euo pipefail

cd ~/low-resource-asr-benchmark
source /home/mila/a/ahomagnr/venvs/asr-benchmark-clean/bin/activate

export HF_DATASETS_CACHE="/network/scratch/a/ahomagnr/hf_cache/datasets/${SLURM_JOB_ID}"
mkdir -p "$HF_DATASETS_CACHE"

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

DATASET="/network/scratch/a/ahomagnr/DDD/DDD-Kenya-Fleurs/Kamba"
OUTPUT="/network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/afrihubert_large/fleurs/kamba/run_001"

mkdir -p "$OUTPUT"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m src.finetuning.afrihubert_train \
    --config configs/afrihubert_large.yaml \
    --dataset-name "$DATASET" \
    --output-dir "$OUTPUT"

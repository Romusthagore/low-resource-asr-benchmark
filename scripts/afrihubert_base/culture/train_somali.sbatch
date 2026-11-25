#!/bin/bash
#SBATCH --job-name=afrihubert-base-somali-culture
#SBATCH --partition=long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=logs/afrihubert-somali-%j.out
#SBATCH --error=logs/afrihubert-somali-%j.err

set -euo pipefail

cd ~/low-resource-asr-benchmark
source /home/mila/a/ahomagnr/venvs/asr-benchmark-clean/bin/activate
export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib:${LD_LIBRARY_PATH:-}"

export HF_DATASETS_CACHE="/network/scratch/a/ahomagnr/hf_cache/datasets/${SLURM_JOB_ID}"
mkdir -p "$HF_DATASETS_CACHE"
trap 'rm -rf "$HF_DATASETS_CACHE"' EXIT

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

DATASET="/network/scratch/a/ahomagnr/DDD/DDD-Kenya-Culture/Somali"
OUTPUT="/network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/afrihubert_base/culture/somali/run_001"

mkdir -p "$OUTPUT"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m src.finetuning.afrihubert_train \
    --config configs/afrihubert_base.yaml \
    --dataset-name "$DATASET" \
    --output-dir "$OUTPUT"

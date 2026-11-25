#!/bin/bash
#SBATCH --job-name=afrihubert-xlarge-gusii
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/afrihubert-xlarge-gusii-%j.out
#SBATCH --error=logs/afrihubert-xlarge-gusii-%j.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="/network/scratch/a/ahomagnr/DDD/DDD-Kenya-Fleurs/Gusii"
OUTPUT="/network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/afrihubert_xlarge/fleurs/gusii/run_001"

mkdir -p "$OUTPUT"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m src.finetuning.afrihubert_train \
    --config configs/afrihubert_xlarge.yaml \
    --dataset-name "$DATASET" \
    --output-dir "$OUTPUT"

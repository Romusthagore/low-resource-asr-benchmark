#!/bin/bash
#SBATCH --job-name=mms-luhya
#SBATCH --partition=long
#SBATCH --constraint=turing
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mms-luhya-%j.out
#SBATCH --error=logs/mms-luhya-%j.err

set -euo pipefail

cd ~/low-resource-asr-benchmark

source /home/mila/a/ahomagnr/venvs/asr-benchmark-clean/bin/activate
export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

export HF_HOME=/network/scratch/a/ahomagnr/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets/${SLURM_JOB_ID}
export HF_HUB_CACHE=$HF_HOME/hub
export TMPDIR=/network/scratch/a/ahomagnr/tmp

mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TMPDIR"
trap 'rm -rf "$HF_DATASETS_CACHE"' EXIT
mkdir -p logs

OUTPUT="/network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/mms/fleurs/luhya/run_001"
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
    --dataset-config Luhya \
    --output-dir "$OUTPUT"

echo "Training completed: $(date)"

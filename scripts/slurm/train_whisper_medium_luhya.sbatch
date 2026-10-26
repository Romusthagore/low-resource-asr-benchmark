#!/bin/bash
#SBATCH --job-name=whisper-medium-luhya
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/whisper_medium_luhya_%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/whisper_medium_luhya_%j.err

set -euo pipefail

cd "$HOME/low-resource-asr-benchmark"
source "$HOME/venvs/asr-benchmark-clean/bin/activate"

export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_EVALUATE_CACHE="${SLURM_TMPDIR:-/tmp}/evaluate-${SLURM_JOB_ID}"
mkdir -p "$HF_EVALUATE_CACHE"
export HF_HOME="/network/scratch/a/ahomagnr/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="/network/scratch/a/ahomagnr/hf-cache/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

# Sanity check: make sure the Whisper-medium checkpoint is present in the HF cache
if ! ls "$HF_HUB_CACHE" 2>/dev/null | grep -qi 'whisper-medium'; then
    echo "ERROR: openai/whisper-medium not found in $HF_HUB_CACHE" >&2
    exit 1
fi
echo "Whisper-medium checkpoint found in HF cache"

WORK_DIR=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/whisper-medium-luhya-representative_50h_seed42
mkdir -p "$WORK_DIR"

srun python -m african_asr.finetuning.whisper_train \
  --config configs/whisper_medium_luhya.yaml

echo "End: $(date)"

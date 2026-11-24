#!/bin/bash
#SBATCH --job-name=whisper-luhya
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/whisper_%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/whisper_%j.err

set -euo pipefail

cd "$HOME/low-resource-asr-benchmark"

source "$HOME/venvs/asr-benchmark-clean/bin/activate"

# Configuration choisie lors de la soumission
CONFIG="${1:?Usage: sbatch scripts/slurm/train_whisper.sbatch configs/whisper_medium_luhya.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    exit 1
fi

# Python imports
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Hugging Face cache
export HF_HOME="/network/scratch/a/ahomagnr/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="/network/scratch/a/ahomagnr/hf-cache/datasets"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Node-local evaluation cache
export HF_EVALUATE_CACHE="${SLURM_TMPDIR:-/tmp}/evaluate-${SLURM_JOB_ID}"
mkdir -p "$HF_EVALUATE_CACHE"

# Preserve library configuration used by previous Mila runs
export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "========================================"
echo "Whisper ASR Fine-Tuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "Start: $(date)"
echo "========================================"

nvidia-smi

# Run fine-tuning
srun python -u -m african_asr.finetuning.whisper_train \
    --config "$CONFIG"

echo "End: $(date)"

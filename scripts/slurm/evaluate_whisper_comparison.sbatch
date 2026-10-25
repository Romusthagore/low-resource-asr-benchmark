#!/bin/bash
#SBATCH --job-name=whisper-compare
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/whisper_compare_%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/whisper_compare_%j.err

set -euo pipefail

# ============================================================
# Environment
# ============================================================
PROJECT="$HOME/low-resource-asr-benchmark"
SCRATCH="/network/scratch/a/ahomagnr/low-resource-asr-benchmark"
VENV="/home/mila/a/ahomagnr/venvs/asr-benchmark-clean"

mkdir -p "$SCRATCH/logs"

source "$VENV/bin/activate"

export HF_HOME="/network/scratch/a/ahomagnr/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="/network/scratch/a/ahomagnr/hf-cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"

# ============================================================
# Dataset input configuration
# ============================================================
DATASET_NAME="Digital-Divide-Data/Luhya-ASR-Data-subset-642H"
TRAIN_SPLIT="train"
TEXT_COLUMN="transcripts"
AUDIO_COLUMN="audio"

TARGET_TRAIN_HOURS=50
SEED=42
VALIDATION_SPLIT_PCT=0.1
TEST_SPLIT_PCT=0.1

# 0 = evaluate the entire test split; use a small number for a test run
MAX_SAMPLES=0

# ============================================================
# Model configuration
# ============================================================
ZERO_SHOT_SMALL="openai/whisper-small"
ZERO_SHOT_MEDIUM="openai/whisper-medium"
ZERO_SHOT_LARGE_V3="openai/whisper-large-v3"

FINE_TUNED_SMALL="$SCRATCH/outputs/whisper-small_luhya_642h_50h_seed42/final_model"
FINE_TUNED_MEDIUM="$SCRATCH/outputs/whisper-medium-luhya_642h_50h_seed42/checkpoint-5600"
FINE_TUNED_LARGE_V3="$SCRATCH/outputs/whisper-large-v3-luhya_642h_50h_seed42/checkpoint-1600"

# ============================================================
# Output configuration
# ============================================================
OUTPUT_DIR="$SCRATCH/whisper_comparison_seed${SEED}_${TARGET_TRAIN_HOURS}h"
METRICS_SCRIPT="$PROJECT/src/african_asr/evaluation/compute_asr_metrics.py"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Whisper zero-shot vs fine-tuned comparison"
echo "Job ID: ${SLURM_JOB_ID:-not-submitted}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Dataset: $DATASET_NAME"
echo "Training subset: ${TARGET_TRAIN_HOURS} hours"
echo "Seed: $SEED"
echo "Max test samples: $MAX_SAMPLES (0 = all)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

nvidia-smi

# Check all fine-tuned checkpoints before starting
for checkpoint in \
    "$FINE_TUNED_SMALL" \
    "$FINE_TUNED_MEDIUM" \
    "$FINE_TUNED_LARGE_V3"
do
    if [[ ! -d "$checkpoint" ]]; then
        echo "ERROR: Fine-tuned checkpoint not found: $checkpoint" >&2
        exit 1
    fi
done

srun python -u scripts/evaluate_whisper_comparison.py \
    --target-train-hours "$TARGET_TRAIN_HOURS" \
    --seed "$SEED" \
    --max-samples "$MAX_SAMPLES"

echo "============================================================"
echo "Evaluation finished: $(date)"
echo "Results:"
echo "  $OUTPUT_DIR/model_comparison.csv"
echo "  $OUTPUT_DIR/model_comparison.json"
echo "============================================================"

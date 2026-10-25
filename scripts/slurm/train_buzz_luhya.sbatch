#!/bin/bash
#SBATCH --job-name=buzz-luhya
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/buzz_luhya_%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/buzz_luhya_%j.err

set -euo pipefail

cd "$HOME/low-resource-asr-benchmark"
source "$HOME/venvs/asr-benchmark-clean/bin/activate"

export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/network/scratch/a/ahomagnr/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

srun "$VIRTUAL_ENV/bin/python" src/african_asr/finetuning/Buzz_train.py \
  --config configs/buzz_luhya.yaml \
  --resume_from_checkpoint /network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/buzzasr-swahili-luhya_642h_50h_seed42/checkpoint-1800

echo "End: $(date)"

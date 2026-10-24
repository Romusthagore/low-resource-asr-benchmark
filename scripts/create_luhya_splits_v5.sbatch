#!/bin/bash
#SBATCH --job-name=luhya-v5
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/luhya-v5-%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/luhya-v5-%j.err

set -euo pipefail

cd /home/mila/a/ahomagnr/low-resource-asr-benchmark

/home/mila/a/ahomagnr/venvs/asr-benchmark-clean/bin/python -u \
  scripts/create_luhya_splits_v5.py \
  --input-dir /network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/s3qa_luhya_full \
  --output-dir /network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/luhya_fixed_splits_seed42_v5 \
  --seed 42 \
  --val-hours 5 \
  --test-hours 10 \
  --attempts 200 \
  --min-speakers 12 \
  --min-gender-speakers 3 \
  --tolerance-seconds 30

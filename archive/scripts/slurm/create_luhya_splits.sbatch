#!/bin/bash
#SBATCH --job-name=luhya-splits
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/luhya-splits-%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/luhya-splits-%j.err

set -euo pipefail

cd /home/mila/a/ahomagnr/low-resource-asr-benchmark

/home/mila/a/ahomagnr/venvs/asr-benchmark-clean/bin/python -u \
  scripts/create_luhya_splits.py \
  --input-dir /network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/s3qa_luhya_full \
  --output-dir /network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/luhya_fixed_splits_seed42 \
  --seed 42 \
  --val-hours 5 \
  --test-hours 10 \
  --attempts 200 \
  --min-speakers 12 \
  --min-gender-speakers 3 \
  --tolerance-seconds 30

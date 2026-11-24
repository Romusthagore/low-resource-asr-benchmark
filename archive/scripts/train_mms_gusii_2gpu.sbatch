#!/bin/bash
#SBATCH --job-name=mms-gusii-2gpu
#SBATCH --partition=long
#SBATCH --constraint=turing
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mms-gusii-2gpu-%j.out
#SBATCH --error=logs/mms-gusii-2gpu-%j.err

set -euo pipefail

cd ~/low-resource-asr-benchmark
source /home/mila/a/ahomagnr/venvs/asr-benchmark-clean/bin/activate

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

echo "=== MMS-1B Gusii Adapter Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

nvidia-smi

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m src.finetuning.mms_train \
    --config "${CONFIG_PATH:-configs/ddp/mms_gusii_2gpu.yaml}"

echo "Training completed: $(date)"

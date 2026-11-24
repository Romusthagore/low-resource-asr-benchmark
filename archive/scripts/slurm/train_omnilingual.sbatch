#!/bin/bash
#SBATCH --job-name=omni-luhya
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/omni_luhya_%j.out
#SBATCH --error=/network/scratch/a/ahomagnr/low-resource-asr-benchmark/logs/omni_luhya_%j.err

set -euo pipefail

cd "$HOME/low-resource-asr-benchmark"
source "$HOME/venvs/omnilingual-asr/bin/activate"

export LD_LIBRARY_PATH="/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/network/scratch/a/ahomagnr/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FAIRSEQ2_CACHE_DIR="/network/scratch/a/ahomagnr/fairseq2"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

# Vérification que le checkpoint Omnilingual est bien dans le cache fairseq2
if ! find "$FAIRSEQ2_CACHE_DIR/assets" -iname 'omniASR-CTC-1B-v2.pt' 2>/dev/null | grep -q .; then
    echo "ERREUR: checkpoint omniASR-CTC-1B-v2.pt introuvable dans $FAIRSEQ2_CACHE_DIR/assets" >&2
    exit 1
fi
echo "Checkpoint Omnilingual trouvé dans le cache fairseq2"

srun python src/african_asr/finetuning/omnilingual_train.py \
  --config configs/omnilingual_luhya.yaml \
  --work-dir /network/scratch/a/ahomagnr/low-resource-asr-benchmark/outputs/omnilingual-ctc-1b-luhya-run2

echo "End: $(date)"

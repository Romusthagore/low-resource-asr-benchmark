
# Stage 1 — Luhya Dataset Preparation

## Overview

This stage prepares the Luhya ASR dataset through acoustic quality assessment and reproducible speaker-disjoint data splitting.

**Dataset:** `Digital-Divide-Data/Luhya-ASR-Data-subset-642H`

## 1. Acoustic Quality Assessment

We use the pretrained S3QA model to assess speech recording quality.

- **Source:** `external/S3QA/` (Git submodule)
- **Script:** `scripts/evaluate_s3qa_full_dataset.py`
- **Checkpoint:** `BEST.ckpt`
- **Audio:** 16 kHz, 4-second windows, 1-second hop
- **Output:** 367 CSV files (201,768 recordings)

### Environment

S3QA uses a dedicated Python environment.

```bash
python3 -m venv s3qa-eval
source s3qa-eval/bin/activate
python -m pip install -r environments/s3qa-requirements.txt
```

## 2. Speaker-Disjoint Splits

**Script:** `scripts/create_luhya_splits.py`

**Random seed:** `42`

| Split | Recordings | Speakers | Hours |
|---|---:|---:|---:|
| Train pool | 181,413 | 179 | 505.13 |
| Validation | 2,725 | 12 | 4.99 |
| Test | 6,226 | 12 | 9.99 |

No speaker or recording overlap exists between splits.

**Output:** `outputs/luhya_fixed_splits_seed42/`

Training subsets (e.g., 50 hours) are selected deterministically using `src/african_asr/data/split_loader.py`.

## Reproducibility

Reproduction requires the pinned S3QA source, checkpoint, dataset revision, environment dependencies, and fixed split manifests.


"""Generic dataset loader with automatic split detection.

Loads a dataset and returns only the splits that actually exist. Handles
three common layouts:

1. A directory of flat manifest files at the top level:
   train.csv, dev.csv / validation.csv, test.csv, ...

2. A Hugging Face dataset repo clone with a `data/` subfolder containing
   per-split parquet shards (e.g. train-00000.parquet, dev-00001.parquet,
   test-00002.parquet). This is the layout produced by the HF dataset
   viewer's "auto-converted parquet" export, and it already embeds audio
   bytes alongside the transcription, so no separate audio archive needs
   to be extracted.

3. A Hugging Face Hub repo id (e.g. "google/WaxalNLP"), loaded directly
   via `datasets.load_dataset`.

- if only "train" exists -> {"train": ...}
- if train/dev/test (or train/validation/test) exist -> all three,
  regardless of whether the source called the middle split "dev"
  or "validation".
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from datasets import Audio, Dataset, DatasetDict, load_dataset

# Canonical split name -> accepted aliases (case-insensitive)
_SPLIT_ALIASES = {
    "train": {"train", "training"},
    "validation": {"validation", "valid", "dev", "development"},
    "test": {"test", "eval", "evaluation"},
}

_ALL_ALIASES = {a for aliases in _SPLIT_ALIASES.values() for a in aliases}

_MANIFEST_EXTENSIONS = (".csv", ".tsv", ".json", ".jsonl", ".parquet")


def _canonical_split_name(name: str) -> str:
    """Map a raw split name to its canonical form if recognized."""
    lower = name.lower()
    for canon, aliases in _SPLIT_ALIASES.items():
        if lower in aliases:
            return canon
    return name  # unknown split name: keep as-is


def _is_local_path(dataset_id: str) -> bool:
    return os.path.exists(dataset_id)


def _find_parquet_shard_dir(path: str) -> Optional[str]:
    """Look for a directory containing per-split parquet shards.

    Checks `path` itself and `path/data` for files named like
    `train-00000.parquet`, `dev-00001.parquet`, `test-00002.parquet`.
    Returns that directory's path, or None if no such layout is found.
    """
    candidates = [path, os.path.join(path, "data")]
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        for f in os.listdir(cand):
            if not f.lower().endswith(".parquet"):
                continue
            stem_prefix = f.lower().split("-", 1)[0].split("_", 1)[0]
            if stem_prefix in _ALL_ALIASES:
                return cand
    return None


def _detect_local_splits(data_dir: str) -> Dict[str, str]:
    """Scan a local directory and map each found split to its file.

    Recognizes filenames like train.csv, dev.tsv, validation.json,
    test.parquet, etc. (single-file-per-split manifests, not shards).
    """
    data_dir = Path(data_dir)
    found: Dict[str, str] = {}

    for f in sorted(data_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _MANIFEST_EXTENSIONS:
            continue
        stem = f.stem.lower()
        if stem not in _ALL_ALIASES:
            continue
        canon = _canonical_split_name(stem)
        found[canon] = str(f)

    return found


def _hf_loader_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext in (".json", ".jsonl"):
        return "json"
    if ext == ".parquet":
        return "parquet"
    raise ValueError(f"Unsupported manifest extension: {ext}")


def load_dataset_with_splits(
    dataset_id: str,
    audio_column: Optional[str] = "audio",
    sampling_rate: int = 16000,
    hf_config: Optional[str] = None,
    **load_kwargs,
) -> DatasetDict:
    """Load a dataset and only the splits that actually exist.

    Parameters
    ----------
    dataset_id:
        Either a Hugging Face Hub repo id (e.g. "google/WaxalNLP"),
        OR a local directory. The local directory can be:
          - a flat folder of manifests (train.csv, dev.csv, test.csv, ...)
          - a cloned HF dataset repo with a `data/` subfolder of
            per-split parquet shards (train-00000.parquet, ...)
    audio_column:
        Name of the audio column to cast to the target sampling rate.
        Set to None to disable audio casting.
    sampling_rate:
        Target sampling rate for the audio column.
    hf_config:
        Optional HF dataset config/subset name (e.g. a language code),
        forwarded to `load_dataset` when loading from the Hub.
    **load_kwargs:
        Extra keyword arguments forwarded to `datasets.load_dataset`.

    Returns
    -------
    A DatasetDict containing only the splits that are present. If only
    "train" exists, the returned dict has a single "train" key. If
    train/validation/test all exist (regardless of whether the source
    called the middle split "dev" or "validation"), all three are
    present under the canonical keys "train", "validation", "test".
    """

    if _is_local_path(dataset_id):
        shard_dir = _find_parquet_shard_dir(dataset_id)
        if shard_dir is not None:
            # Richest layout: parquet shards already embed audio + text.
            # `datasets` auto-detects splits from the "train-", "dev-",
            # "test-" filename prefixes.
            loaded = load_dataset("parquet", data_dir=shard_dir, **load_kwargs)
            datasets_dict = {
                _canonical_split_name(name): ds for name, ds in loaded.items()
            }
        else:
            split_files = _detect_local_splits(dataset_id)
            if not split_files:
                raise FileNotFoundError(
                    f"No recognizable split layout (parquet shards under "
                    f"'data/', or flat train/dev/validation/test manifests) "
                    f"found in {dataset_id!r}."
                )

            datasets_dict: Dict[str, Dataset] = {}
            for split_name, path in split_files.items():
                fmt = _hf_loader_format(path)
                ds = load_dataset(fmt, data_files=path, split="train", **load_kwargs)
                datasets_dict[split_name] = ds

    else:
        # Hub dataset: let `load_dataset` return whatever splits exist.
        try:
            loaded = load_dataset(dataset_id, hf_config, **load_kwargs)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load dataset '{dataset_id}' from the Hub."
            ) from e

        if isinstance(loaded, DatasetDict):
            datasets_dict = {
                _canonical_split_name(name): ds for name, ds in loaded.items()
            }
        else:
            # A single split was returned (e.g. only "train" exists upstream).
            datasets_dict = {"train": loaded}

    result = DatasetDict(datasets_dict)

    if audio_column is not None:
        for split in result:
            if audio_column in result[split].column_names:
                result[split] = result[split].cast_column(
                    audio_column, Audio(sampling_rate=sampling_rate)
                )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test loading a dataset with its splits.")
    parser.add_argument("dataset_id", help="Local directory or HF Hub repo id")
    parser.add_argument("--hf_config", default=None, help="Optional HF config/subset")
    parser.add_argument("--no_audio_cast", action="store_true", help="Disable audio casting")
    args = parser.parse_args()

    dd = load_dataset_with_splits(
        os.path.expanduser(args.dataset_id),
        audio_column=None if args.no_audio_cast else "audio",
        hf_config=args.hf_config,
    )

    print(f"Splits found: {list(dd.keys())}")
    for split, ds in dd.items():
        print(f"  {split}: {len(ds)} examples — columns: {ds.column_names}")

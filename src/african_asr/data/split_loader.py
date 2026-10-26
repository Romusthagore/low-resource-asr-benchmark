"""Reproducible Luhya ASR loader with configurable training hours."""

from pathlib import Path
import hashlib
import random

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets


DEFAULT_SPLIT_DIR = Path(
    "/network/scratch/a/ahomagnr/"
    "low-resource-asr-benchmark/outputs/luhya_representative_splits_seed42"
)

DEFAULT_ARROW_ROOT = Path(
    "/network/scratch/a/ahomagnr/huggingface/hub/"
    "datasets--Digital-Divide-Data--Luhya-ASR-Data-subset-642H"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_train_hours(df, target_hours=50, seed=42):
    """Select a deterministic, nested subset from the training pool."""

    if target_hours is None:
        return df.copy().reset_index(drop=True)

    target_hours = float(target_hours)

    if target_hours <= 0:
        raise ValueError("target_train_hours must be positive")

    durations = pd.to_numeric(
        df["duration_seconds"], errors="raise"
    )

    if durations.isna().any() or (durations <= 0).any():
        raise ValueError("Invalid training durations")

    indices = list(range(len(df)))
    random.Random(int(seed)).shuffle(indices)

    selected = []
    total_seconds = 0.0
    target_seconds = target_hours * 3600

    for idx in indices:
        selected.append(idx)
        total_seconds += float(durations.iloc[idx])

        if total_seconds >= target_seconds:
            break

    if total_seconds < target_seconds:
        raise ValueError(
            f"Requested {target_hours}h, "
            f"but only {total_seconds / 3600:.3f}h available"
        )

    return df.iloc[selected].copy().reset_index(drop=True)


def load_manifests(
    split_dir=DEFAULT_SPLIT_DIR,
    target_train_hours=50,
    seed=42,
):
    """Load fixed evaluation splits and dynamically select training data."""

    split_dir = Path(split_dir)

    paths = {
        "train": split_dir / "train_pool.csv",
        "validation": split_dir / "validation_5h.csv",
        "test": split_dir / "test_10h.csv",
    }

    frames = {}

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        frames[name] = pd.read_csv(path, low_memory=False)

    # Select training subset only.
    frames["train"] = select_train_hours(
        frames["train"],
        target_hours=target_train_hours,
        seed=seed,
    )

    required = {
        "arrow_file",
        "row_index",
        "duration_seconds",
        "speaker_name",
    }

    recording_sets = {}
    speaker_sets = {}

    for name, df in frames.items():
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name}: missing columns {sorted(missing)}"
            )

        if df[list(required)].isna().any().any():
            raise ValueError(f"{name}: null required values")

        if df.duplicated(["arrow_file", "row_index"]).any():
            raise ValueError(f"{name}: duplicate recordings")

        recording_sets[name] = set(
            zip(
                df["arrow_file"].astype(str),
                df["row_index"].astype(int),
            )
        )

        speaker_sets[name] = set(
            df["speaker_name"].astype(str)
        )

        hours = df["duration_seconds"].sum() / 3600

        print(
            f"[{name}] {len(df):,} recordings | "
            f"{hours:.3f} hours"
        )

    for a, b in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if recording_sets[a] & recording_sets[b]:
            raise ValueError(f"Recording overlap: {a}/{b}")


    print("[OK] No recording overlap (speaker overlap allowed)")

    # Verify the original frozen evaluation manifests.
    for filename in ("validation_5h.csv", "test_10h.csv"):
        print(
            f"[SHA256] {filename}: "
            f"{sha256_file(split_dir / filename)}"
        )

    return frames


def materialize_arrow_split(
    manifest,
    arrow_root=DEFAULT_ARROW_ROOT,
    audio_column="audio",
    text_column="transcripts",
    sampling_rate=16000,
):
    """Resolve Arrow references into an ASR-compatible Dataset."""

    arrow_root = Path(arrow_root)
    pieces = []

    for relative_path, group in manifest.groupby(
        "arrow_file", sort=False
    ):
        path = arrow_root / relative_path

        if not path.is_file():
            raise FileNotFoundError(path)

        source = Dataset.from_file(str(path))
        indices = group["row_index"].astype(int).tolist()

        if any(i < 0 or i >= len(source) for i in indices):
            raise IndexError(f"Invalid row_index in {path}")

        subset = source.select(indices)

        if text_column not in subset.column_names:
            raise KeyError(f"Missing '{text_column}' in {path}")

        if audio_column not in subset.column_names:
            raise KeyError(f"Missing '{audio_column}' in {path}")

        subset = subset.select_columns(
            [audio_column, text_column]
        )

        if text_column != "sentence":
            subset = subset.rename_column(
                text_column, "sentence"
            )

        pieces.append(subset)

    if not pieces:
        raise ValueError("Empty split")

    dataset = concatenate_datasets(pieces)

    return dataset.cast_column(
        audio_column,
        Audio(sampling_rate=sampling_rate),
    )


def load_fixed_luhya_splits(
    split_dir=DEFAULT_SPLIT_DIR,
    arrow_root=DEFAULT_ARROW_ROOT,
    target_train_hours=50,
    seed=42,
    sampling_rate=16000,
    materialize=True,
):
    """Load common ASR splits with YAML-configurable training hours."""

    manifests = load_manifests(
        split_dir=split_dir,
        target_train_hours=target_train_hours,
        seed=seed,
    )

    if not materialize:
        return manifests

    datasets = {}

    for name, manifest in manifests.items():
        datasets[name] = materialize_arrow_split(
            manifest,
            arrow_root=arrow_root,
            sampling_rate=sampling_rate,
        )
        print(f"[loaded] {name}: {len(datasets[name]):,} recordings")

    return DatasetDict(datasets)


if __name__ == "__main__":
    load_fixed_luhya_splits(materialize=False)

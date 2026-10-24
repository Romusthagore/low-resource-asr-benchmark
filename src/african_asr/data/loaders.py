"""Dataset loading and reproducible duration-based splitting utilities."""

import io
import random

import soundfile as sf
from datasets import Audio, DatasetDict, load_dataset


def _audio_duration_seconds(audio):
    """Read audio duration from its bytes or accessible file path."""
    if not isinstance(audio, dict):
        raise ValueError(f"Expected an audio dictionary, got {type(audio).__name__}")

    data = audio.get("bytes")
    path = audio.get("path")

    if data is not None:
        info = sf.info(io.BytesIO(data))
    elif path:
        info = sf.info(path)
    else:
        raise ValueError("Audio example has neither usable bytes nor a path.")

    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate: {info.samplerate}")

    return info.frames / info.samplerate


def _select_by_hours(dataset, target_hours, audio_column="audio", seed=42):
    """Select a reproducible subset reaching a target duration."""
    dataset = dataset.cast_column(audio_column, Audio(decode=False))
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    chosen = []
    seconds = 0.0
    target_seconds = target_hours * 3600

    for idx in indices:
        audio = dataset[idx][audio_column]
        seconds += _audio_duration_seconds(audio)
        chosen.append(idx)
        if seconds >= target_seconds:
            break

    if seconds < target_seconds:
        raise ValueError(
            f"Dataset contains only {seconds / 3600:.3f} hours; "
            f"target was {target_hours:.3f} hours."
        )

    result = dataset.select(chosen)
    print(
        f"[ASR data] Selected {len(result)} examples, "
        f"{seconds / 3600:.3f} h (target: {target_hours:.3f} h)."
    )
    return result


def _select_splits_by_hours(
    dataset,
    audio_column="audio",
    target_train_hours=50,
    validation_split_pct=0.1,
    test_split_pct=0.1,
    seed=42,
):
    """Build disjoint train/validation/test splits by actual audio duration."""
    val_pct = float(validation_split_pct)
    test_pct = float(test_split_pct)
    train_pct = 1.0 - val_pct - test_pct

    if target_train_hours <= 0:
        raise ValueError("target_train_hours must be positive.")
    if val_pct < 0 or test_pct < 0 or train_pct <= 0:
        raise ValueError("Invalid validation/test split percentages.")

    targets = {
        "train": float(target_train_hours),
        "validation": float(target_train_hours) * val_pct / train_pct,
        "test": float(target_train_hours) * test_pct / train_pct,
    }

    dataset = dataset.cast_column(audio_column, Audio(decode=False))
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    selected = {name: [] for name in targets}
    durations = {name: 0.0 for name in targets}
    split_names = list(targets)
    current = 0

    # Read one audio at a time. Never decode the entire column at once.
    for idx in indices:
        if current >= len(split_names):
            break

        name = split_names[current]
        audio = dataset[idx][audio_column]
        duration = _audio_duration_seconds(audio)

        selected[name].append(idx)
        durations[name] += duration

        if durations[name] >= targets[name] * 3600:
            current += 1

    if current < len(split_names):
        remaining = split_names[current:]
        raise ValueError(
            f"Dataset exhausted before reaching duration targets: {remaining}"
        )

    result = DatasetDict({
        name: dataset.select(selected[name])
        for name in split_names
    })

    print("\n[ASR data] Duration-based split selection:")
    for name in split_names:
        print(
            f"  {name:10s}: {len(result[name]):6d} examples | "
            f"{durations[name] / 3600:.3f} h "
            f"(target: {targets[name]:.3f} h)"
        )

    return result


def load_asr_dataset(
    train_csv=None,
    eval_csv=None,
    dataset_name="Digital-Divide-Data/Luhya-ASR-Data-subset-642H",
    dataset_config=None,
    train_split="train",
    eval_split=None,
    test_split=None,
    text_column="transcripts",
    audio_column="audio",
    sample=True,
    sample_size=None,
    target_train_hours=None,
    validation_split_pct=0.1,
    test_split_pct=0.1,
    seed=42,
    **kwargs,
):
    """Load ASR data and optionally create duration-controlled splits."""

    if train_csv:
        from datasets import Audio as AudioFeature
        from datasets import load_dataset as hf_load_dataset

        files = {"train": train_csv}
        if eval_csv:
            files["validation"] = eval_csv

        loaded = hf_load_dataset("csv", data_files=files)
        if "validation" not in loaded:
            split_result = loaded["train"].train_test_split(
                test_size=validation_split_pct,
                seed=seed,
            )
            loaded = DatasetDict({
                "train": split_result["train"],
                "validation": split_result["test"],
            })
        dataset_dict = loaded
    else:
        if not dataset_name:
            raise ValueError("Provide dataset_name or train_csv.")

        loaded = load_dataset(
            dataset_name,
            dataset_config,
            **({"split": train_split} if train_split else {}),
        )

        # A single Dataset was loaded when split=train; build all three
        # disjoint splits from that source when duration targets are requested.
        if not isinstance(loaded, DatasetDict):
            if sample and target_train_hours is not None:
                dataset_dict = _select_splits_by_hours(
                    loaded,
                    audio_column=audio_column,
                    target_train_hours=target_train_hours,
                    validation_split_pct=validation_split_pct,
                    test_split_pct=test_split_pct,
                    seed=seed,
                )
            elif sample and sample_size is not None:
                count = min(int(sample_size), len(loaded))
                loaded = loaded.shuffle(seed=seed).select(range(count))
                split_test = loaded.train_test_split(
                    test_size=test_split_pct, seed=seed
                )
                train_val = split_test["train"].train_test_split(
                    test_size=validation_split_pct
                    / (1.0 - test_split_pct),
                    seed=seed,
                )
                dataset_dict = DatasetDict({
                    "train": train_val["train"],
                    "validation": train_val["test"],
                    "test": split_test["test"],
                })
            else:
                # Keep an existing single-split workflow usable.
                dataset_dict = DatasetDict({"train": loaded})
        else:
            # Respect existing official splits if supplied by the dataset.
            dataset_dict = loaded

    # Normalize the transcript column to "sentence".
    normalized = {}
    for split_name, subset in dataset_dict.items():
        if audio_column not in subset.column_names:
            raise ValueError(
                f"Audio column '{audio_column}' missing in split '{split_name}'. "
                f"Available: {subset.column_names}"
            )

        if text_column not in subset.column_names:
            if text_column == "transcripts" and "sentence" in subset.column_names:
                actual_text_column = "sentence"
            else:
                raise ValueError(
                    f"Text column '{text_column}' missing in split '{split_name}'. "
                    f"Available: {subset.column_names}"
                )
        else:
            actual_text_column = text_column

        if actual_text_column != "sentence":
            if "sentence" in subset.column_names:
                subset = subset.remove_columns(["sentence"])
            subset = subset.rename_column(actual_text_column, "sentence")

        # Ensure the audio feature is standardized to 16 kHz for model inputs.
        subset = subset.cast_column(audio_column, Audio(sampling_rate=16000))
        normalized[split_name] = subset

    return DatasetDict(normalized)

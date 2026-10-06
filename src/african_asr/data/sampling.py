"""Sampling utilities for the African ASR benchmark."""

from datasets import Dataset, DatasetDict


def get_audio_duration(example: dict) -> float:
    """Return the duration of an audio example in seconds."""

    audio = example["audio"]

    if "array" in audio and "sampling_rate" in audio:
        return len(audio["array"]) / audio["sampling_rate"]

    if "duration" in audio:
        return float(audio["duration"])

    raise ValueError(
        "Could not determine audio duration from the audio column."
    )


def add_duration_column(
    dataset: Dataset,
    column_name: str = "duration",
) -> Dataset:
    """Add an audio duration column in seconds."""

    return dataset.map(
        lambda example: {
            column_name: get_audio_duration(example)
        },
        desc="Computing audio durations",
    )


def sample_by_duration(
    dataset: Dataset,
    target_hours: float,
    seed: int = 42,
    duration_column: str = "duration",
) -> Dataset:
    """Sample examples until reaching a target amount of audio."""

    if duration_column not in dataset.column_names:
        dataset = add_duration_column(
            dataset,
            column_name=duration_column,
        )

    target_seconds = target_hours * 3600

    shuffled_dataset = dataset.shuffle(seed=seed)

    selected_indices = []
    total_duration = 0.0

    for index, duration in enumerate(
        shuffled_dataset[duration_column]
    ):
        if total_duration >= target_seconds:
            break

        selected_indices.append(index)
        total_duration += float(duration)

    sampled_dataset = shuffled_dataset.select(selected_indices)

    return sampled_dataset


def sample_dataset_dict(
    dataset: DatasetDict,
    target_hours: float,
    split: str = "train",
    seed: int = 42,
) -> DatasetDict:
    """Sample one split of a DatasetDict by target audio duration."""

    if split not in dataset:
        raise ValueError(
            f"Split '{split}' not found. "
            f"Available splits: {list(dataset.keys())}"
        )

    sampled_split = sample_by_duration(
        dataset[split],
        target_hours=target_hours,
        seed=seed,
    )

    result = DatasetDict(dataset)
    result[split] = sampled_split

    return result

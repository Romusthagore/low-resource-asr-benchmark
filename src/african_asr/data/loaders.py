"""Dataset loading utilities for the African ASR benchmark."""

from datasets import Dataset, DatasetDict, load_dataset


AUDIO_COLUMN_CANDIDATES = [
    "audio",
    "Audio",
    "audio_filepath",
    "audio_file",
    "audio_path",
    "wav",
    "wav_file",
    "path",
]

TEXT_COLUMN_CANDIDATES = [
    "text",
    "texts",
    "Text",
    "sentence",
    "transcription",
    "transcript",
    "transcripts",
    "transcription_text",
    "utterance",
    "normalized_text",
]


def load_asr_dataset(
    dataset_name: str,
    config_name: str | None = None,
    split: str | None = None,
) -> DatasetDict:
    """Load an ASR dataset from the Hugging Face Hub."""

    if config_name is not None:
        dataset = load_dataset(
            dataset_name,
            config_name,
            split=split,
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
        )

    if isinstance(dataset, DatasetDict):
        return dataset

    split_name = split or "train"

    return DatasetDict({split_name: dataset})


def find_column(
    dataset: Dataset | DatasetDict,
    candidates: list[str],
) -> str:
    """Find the first matching column from a list of candidates."""

    if isinstance(dataset, Dataset):
        columns = dataset.column_names
    else:
        columns = dataset[next(iter(dataset))].column_names

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Could not find a matching column.\n"
        f"Candidates: {candidates}\n"
        f"Available columns: {columns}"
    )


def detect_asr_columns(
    dataset: Dataset | DatasetDict,
) -> tuple[str, str]:
    """Automatically detect audio and transcription columns."""

    audio_column = find_column(
        dataset,
        AUDIO_COLUMN_CANDIDATES,
    )

    text_column = find_column(
        dataset,
        TEXT_COLUMN_CANDIDATES,
    )

    return audio_column, text_column


def standardize_columns(
    dataset: Dataset | DatasetDict,
) -> Dataset | DatasetDict:
    """Standardize ASR column names to ``audio`` and ``text``."""

    audio_column, text_column = detect_asr_columns(dataset)

    rename_map = {}

    if audio_column != "audio":
        rename_map[audio_column] = "audio"

    if text_column != "text":
        rename_map[text_column] = "text"

    if rename_map:
        dataset = dataset.rename_columns(rename_map)

    return dataset


def load_and_standardize_asr_dataset(
    dataset_name: str,
    config_name: str | None = None,
    split: str | None = None,
) -> DatasetDict:
    """Load an ASR dataset and standardize its columns."""

    dataset = load_asr_dataset(
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
    )

    return standardize_columns(dataset)

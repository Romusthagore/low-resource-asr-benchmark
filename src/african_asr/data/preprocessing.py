"""Audio and text preprocessing utilities for the African ASR benchmark."""

from typing import Any

from datasets import Dataset, DatasetDict, Audio

from .normalization import normalize_text


def cast_audio_column(
    dataset: Dataset | DatasetDict,
    sampling_rate: int = 16_000,
) -> Dataset | DatasetDict:
    """Cast the audio column to a standardized sampling rate."""

    return dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate),
    )


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    unicode_normalize: str | None = "NFC",
) -> str:
    """Normalize an ASR transcription."""

    return normalize_text(
        text=text,
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        unicode_normalize=unicode_normalize,
    )


def add_normalized_text(
    dataset: Dataset | DatasetDict,
    output_column: str = "normalized_text",
    **normalization_kwargs: Any,
) -> Dataset | DatasetDict:
    """Add a normalized transcription column."""

    def normalize_example(example: dict[str, Any]) -> dict[str, str]:
        return {
            output_column: preprocess_text(
                example["text"],
                **normalization_kwargs,
            )
        }

    return dataset.map(
        normalize_example,
        desc="Normalizing transcriptions",
    )


def preprocess_asr_dataset(
    dataset: Dataset | DatasetDict,
    sampling_rate: int = 16_000,
    add_normalized_transcripts: bool = True,
    **normalization_kwargs: Any,
) -> Dataset | DatasetDict:
    """Apply the standard benchmark preprocessing pipeline."""

    dataset = cast_audio_column(
        dataset,
        sampling_rate=sampling_rate,
    )

    if add_normalized_transcripts:
        dataset = add_normalized_text(
            dataset,
            **normalization_kwargs,
        )

    return dataset

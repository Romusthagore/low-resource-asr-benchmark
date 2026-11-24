"""Shared preprocessing for CTC ASR (Wav2Vec2-BERT, AfriHuBERT, MMS).

Assumes splits have already been loaded by src.data.loaders and audio cast
at the required sample rate. Does not create or alter dataset splits.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from datasets import Dataset


def normalize_transcription(
    text: str,
    *,
    lowercase: bool = True,
    allowed_characters: Optional[str] = None,
    strip_diacritics: bool = False,
) -> str:
    """Normalize text without silently stripping African-language diacritics.

    Set allowed_characters only when an explicit language-specific alphabet
    has been established. A literal space is preserved if allowed.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("ʼ", "'").replace("’", "'")
    if lowercase:
        text = text.lower()
    if strip_diacritics:
        text = "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
        text = unicodedata.normalize("NFC", text)
    if allowed_characters is not None:
        allowed = set(allowed_characters)
        text = "".join(c for c in text if c in allowed)
    return re.sub(r"\s+", " ", text).strip()


def prepare_ctc_dataset(
    dataset: Dataset,
    processor,
    *,
    text_column: str = "transcripts",
    audio_column: str = "audio",
    max_audio_length: Optional[float] = 30.0,
    lowercase: bool = True,
    allowed_characters: Optional[str] = None,
    strip_diacritics: bool = False,
    batch_size: int = 16,
    num_proc: Optional[int] = None,
    desc: Optional[str] = None,
) -> Dataset:
    """Filter invalid examples and encode audio + CTC labels.

    Returns a dataset with input_values OR input_features, labels and length.
    The same normalization options must be used across all official splits.
    """
    if audio_column not in dataset.column_names:
        raise KeyError(f"Audio column {audio_column!r} not found: {dataset.column_names}")
    if text_column not in dataset.column_names:
        raise KeyError(f"Text column {text_column!r} not found: {dataset.column_names}")

    def valid_example(example):
        try:
            audio = example[audio_column]
            waveform = audio["array"]
            sr = audio["sampling_rate"]
            if sr <= 0 or waveform is None or len(waveform) == 0:
                return False
            if max_audio_length is not None and len(waveform) / sr > max_audio_length:
                return False
            cleaned = normalize_transcription(
                example[text_column], lowercase=lowercase,
                allowed_characters=allowed_characters,
                strip_diacritics=strip_diacritics,
            )
            return bool(cleaned)
        except (KeyError, TypeError, ValueError, OSError):
            return False

    dataset = dataset.filter(
        valid_example,
        num_proc=num_proc,
        desc="Filter invalid CTC examples",
    )

    def encode_batch(batch):
        audios = batch[audio_column]
        sampling_rates = {a["sampling_rate"] for a in audios}
        if len(sampling_rates) != 1:
            raise ValueError(f"Mixed sampling rates in batch: {sampling_rates}")

        audio_arrays = [a["array"] for a in audios]
        texts = [
            normalize_transcription(
                t, lowercase=lowercase,
                allowed_characters=allowed_characters,
                strip_diacritics=strip_diacritics,
            )
            for t in batch[text_column]
        ]

        audio_features = processor(
            audio_arrays,
            sampling_rate=next(iter(sampling_rates)),
        )

        if "input_features" in audio_features:
            key = "input_features"
        elif "input_values" in audio_features:
            key = "input_values"
        else:
            raise ValueError(
                "Processor returned neither input_features nor input_values"
            )

        labels = processor(text=texts).input_ids
        values = audio_features[key]

        return {
            key: values,
            "labels": labels,
            "length": [len(v) for v in values],
        }

    return dataset.map(
        encode_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=desc or "Encode CTC dataset",
    )

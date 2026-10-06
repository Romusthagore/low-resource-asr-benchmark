"""Dataset loading utilities for the African ASR benchmark."""

from datasets import Audio, DatasetDict, load_dataset


def load_asr_dataset(
    train_csv=None,
    eval_csv=None,
    dataset_name="DDD-Kenya/Luhya-ASR-Data-subset-50h",
    dataset_config=None,
    train_split="train",
    eval_split="validation",
    text_column="transcript",
    audio_column="audio",
    sample=True,
    sample_size=3600,
    validation_split_pct=0.2,
    seed=42,
):
    """
    Load and prepare an ASR dataset.

    This function mirrors the data-loading behavior of the reference
    Whisper fine-tuning pipeline.

    Returns:
        DatasetDict with:
            - "train"
            - "validation"

    Notes:
        - Audio is cast to 16 kHz.
        - The transcript column is standardized to "sentence".
        - Sampling is performed before audio preprocessing.
        - Audio duration is NOT computed here.
    """

    # ------------------------------------------------------------------
    # 1. Local CSV input
    # ------------------------------------------------------------------
    if train_csv:
        data_files = {"train": train_csv}

        if eval_csv:
            data_files["validation"] = eval_csv

        ds = load_dataset(
            "csv",
            data_files=data_files,
        )

        if audio_column != "audio":
            ds = ds.rename_column(
                audio_column,
                "audio",
            )

        if text_column != "sentence":
            ds = ds.rename_column(
                text_column,
                "sentence",
            )

        ds = ds.cast_column(
            "audio",
            Audio(sampling_rate=16000),
        )

        if "validation" not in ds:
            split = ds["train"].train_test_split(
                test_size=validation_split_pct,
                seed=seed,
            )

            ds = DatasetDict(
                train=split["train"],
                validation=split["test"],
            )

        return ds

    # ------------------------------------------------------------------
    # 2. Hugging Face dataset
    # ------------------------------------------------------------------
    if dataset_name:
        ds_all = load_dataset(
            dataset_name,
            dataset_config,
        )

        if eval_split in ds_all:
            train_raw = ds_all[train_split]
            eval_raw = ds_all[eval_split]

        else:
            split = ds_all[train_split].train_test_split(
                test_size=validation_split_pct,
                seed=seed,
            )

            train_raw = split["train"]
            eval_raw = split["test"]

        # --------------------------------------------------------------
        # Sampling
        # --------------------------------------------------------------
        if sample:
            train_raw = train_raw.shuffle(
                seed=seed
            ).select(
                range(
                    min(
                        sample_size,
                        len(train_raw),
                    )
                )
            )

            eval_n = max(
                1,
                int(
                    sample_size * validation_split_pct
                ),
            )

            eval_raw = eval_raw.shuffle(
                seed=seed
            ).select(
                range(
                    min(
                        eval_n,
                        len(eval_raw),
                    )
                )
            )

        ds = DatasetDict(
            train=train_raw,
            validation=eval_raw,
        )

        # --------------------------------------------------------------
        # Standardize column names
        # --------------------------------------------------------------
        if audio_column != "audio":
            ds = ds.rename_column(
                audio_column,
                "audio",
            )

        if text_column != "sentence":
            ds = ds.rename_column(
                text_column,
                "sentence",
            )

        # --------------------------------------------------------------
        # Standardize audio sampling rate
        # --------------------------------------------------------------
        ds = ds.cast_column(
            "audio",
            Audio(sampling_rate=16000),
        )

        return ds

    raise ValueError(
        "Provide either --train_csv (+ optional --eval_csv) "
        "or --dataset_name."
    )

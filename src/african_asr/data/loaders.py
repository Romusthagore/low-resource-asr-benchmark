"""Dataset loading utilities for the African ASR benchmark."""

from datasets import Audio, DatasetDict, load_dataset


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
    validation_split_pct=0.2,
    test_split_pct=0.1,
    seed=42,
):
    """
    Load and prepare an ASR dataset.

    For datasets with only one training split, the data is deterministically
    partitioned into independent train/validation/test subsets.

    Protocol:
        1. Reserve test_split_pct of the full dataset for test.
        2. Split the remaining pool into train and validation using
           validation_split_pct.
        3. If sampling is enabled, select sample_size examples from train.
        4. Validation and test are never sampled by sample_size.

    Returns:
        DatasetDict with:
            - "train"
            - "validation"
            - "test"

    Notes:
        - Audio is cast to 16 kHz.
        - The transcript column is standardized to "sentence".
        - Splits are deterministic given seed.
    """

    # ------------------------------------------------------------------
    # 1. Local CSV input
    # ------------------------------------------------------------------
    if train_csv:
        data_files = {"train": train_csv}

        if eval_csv:
            data_files["validation"] = eval_csv

        ds = load_dataset("csv", data_files=data_files)

        if "validation" not in ds:
            split = ds["train"].train_test_split(
                test_size=validation_split_pct,
                seed=seed,
            )

            ds = DatasetDict(
                train=split["train"],
                validation=split["test"],
            )

        # Standardize column names.
        if audio_column != "audio":
            for split_name in ds:
                ds[split_name] = ds[split_name].rename_column(
                    audio_column,
                    "audio",
                )

        if text_column != "sentence":
            for split_name in ds:
                ds[split_name] = ds[split_name].rename_column(
                    text_column,
                    "sentence",
                )

        ds = ds.cast_column(
            "audio",
            Audio(sampling_rate=16000),
        )

        return ds

    # ------------------------------------------------------------------
    # 2. Hugging Face dataset
    # ------------------------------------------------------------------
    if not dataset_name:
        raise ValueError(
            "Provide either --train_csv (+ optional --eval_csv) "
            "or --dataset_name."
        )

    ds_all = load_dataset(
        dataset_name,
        dataset_config,
    )

    # ------------------------------------------------------------------
    # Case A: official train / validation / test splits exist.
    # ------------------------------------------------------------------
    if (
        eval_split
        and eval_split in ds_all
        and test_split
        and test_split in ds_all
    ):
        train_raw = ds_all[train_split]
        eval_raw = ds_all[eval_split]
        test_raw = ds_all[test_split]

    # ------------------------------------------------------------------
    # Case B: only a train split exists.
    #
    # This is the case for:
    # Digital-Divide-Data/Luhya-ASR-Data-subset-642H
    # ------------------------------------------------------------------
    else:
        full_train = ds_all[train_split]

        if not 0 < test_split_pct < 1:
            raise ValueError(
                "test_split_pct must be strictly between 0 and 1."
            )

        if not 0 < validation_split_pct < 1:
            raise ValueError(
                "validation_split_pct must be strictly between 0 and 1."
            )

        # First isolate the test set.
        test_partition = full_train.train_test_split(
            test_size=test_split_pct,
            seed=seed,
        )

        train_val_pool = test_partition["train"]
        test_raw = test_partition["test"]

        # Then split the remaining pool into train and validation.
        train_val_partition = train_val_pool.train_test_split(
            test_size=validation_split_pct,
            seed=seed,
        )

        train_raw = train_val_partition["train"]
        eval_raw = train_val_partition["test"]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    if sample:
        if sample_size is None:
            raise ValueError(
                "sample_size must be specified when sample=True."
            )

        if sample_size > len(train_raw):
            raise ValueError(
                f"sample_size={sample_size} exceeds the available "
                f"training pool ({len(train_raw)} examples)."
            )

        train_raw = train_raw.shuffle(
            seed=seed,
        ).select(
            range(sample_size)
        )

    # ------------------------------------------------------------------
    # Build benchmark splits.
    # Validation and test remain independent and are never sampled.
    # ------------------------------------------------------------------
    ds = DatasetDict(
        {
            "train": train_raw,
            "validation": eval_raw,
            "test": test_raw,
        }
    )

    # ------------------------------------------------------------------
    # Standardize column names.
    # ------------------------------------------------------------------
    if audio_column != "audio":
        for split_name in ds:
            ds[split_name] = ds[split_name].rename_column(
                audio_column,
                "audio",
            )

    if text_column != "sentence":
        for split_name in ds:
            ds[split_name] = ds[split_name].rename_column(
                text_column,
                "sentence",
            )

    # ------------------------------------------------------------------
    # Standardize audio sampling rate.
    # ------------------------------------------------------------------
    ds = ds.cast_column(
        "audio",
        Audio(sampling_rate=16000),
    )

    return ds

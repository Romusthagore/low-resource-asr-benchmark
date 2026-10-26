from african_asr.data.loaders import load_asr_dataset

print("========== LOADING FIXED SPLITS ==========", flush=True)

dataset = load_asr_dataset(
    dataset_name="Digital-Divide-Data/Luhya-ASR-Data-subset-642H",
    target_train_hours=50,
    seed=42,
    use_fixed_splits=True,
)

print("\n========== DATASET ==========", flush=True)
print(dataset, flush=True)

expected = {
    "train": 17847,
    "validation": 2725,
    "test": 6226,
}

for name, expected_count in expected.items():
    split = dataset[name]

    print(f"\n{name.upper()}", flush=True)
    print("Examples:", len(split), flush=True)
    print("Columns:", split.column_names, flush=True)

    assert len(split) == expected_count
    assert "audio" in split.column_names
    assert "sentence" in split.column_names

    # Lire uniquement la transcription, sans décoder l'audio.
    text_only = split.select([0]).select_columns(["sentence"])
    print("First transcription:", text_only[0]["sentence"], flush=True)

print("\nALL CHECKS PASSED", flush=True)

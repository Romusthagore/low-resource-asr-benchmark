"""Prepare Luhya data and launch the official Omnilingual ASR recipe."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

from african_asr.data.loaders import load_asr_dataset


DEFAULT_CONFIG = "configs/omnilingual_luhya.yaml"
DEFAULT_WORK_DIR = "./outputs/omnilingual-ctc-300m-luhya"
DEFAULT_DATASET = "DDD-Kenya/Luhya-ASR-Data-subset-50h"
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_MAX_AUDIO_LENGTH = 30.0
OMNILINGUAL_ROOT = Path.home() / "omnilingual-asr"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Luhya data and fine-tune Omnilingual CTC."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--sampling-rate", type=int, default=DEFAULT_SAMPLING_RATE)
    parser.add_argument(
        "--max-audio-length",
        type=float,
        default=DEFAULT_MAX_AUDIO_LENGTH,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data/configuration without launching training.",
    )
    return parser.parse_args()


def write_wav(
    path: Path,
    waveform: np.ndarray,
    sampling_rate: int,
):
    """Write a mono float waveform as 16-bit PCM WAV."""

    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim != 1:
        raise ValueError(
            f"Expected mono waveform, got shape {waveform.shape}."
        )

    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sampling_rate)
        wav.writeframes(pcm.tobytes())


def prepare_split(
    dataset,
    split_name: str,
    output_dir: Path,
    sampling_rate: int,
    max_audio_length: float,
):
    """Materialize one split in the official Omnilingual manifest format."""

    audio_dir = output_dir / "audio" / split_name
    manifest_dir = output_dir / "manifest"

    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = manifest_dir / f"{split_name}.tsv"
    wrd_path = manifest_dir / f"{split_name}.wrd"

    kept = 0
    dropped = 0

    with (
        tsv_path.open("w", encoding="utf-8") as tsv,
        wrd_path.open("w", encoding="utf-8") as wrd,
    ):
        tsv.write(f"{audio_dir.resolve()}\n")

        for index, example in enumerate(dataset):
            audio = example["audio"]

            waveform = np.asarray(
                audio["array"],
                dtype=np.float32,
            )

            source_rate = int(audio["sampling_rate"])

            if source_rate != sampling_rate:
                raise ValueError(
                    f"{split_name}[{index}] has sampling rate "
                    f"{source_rate}, expected {sampling_rate}."
                )

            duration = len(waveform) / sampling_rate

            if duration < 1.0 or duration > max_audio_length:
                dropped += 1
                continue

            transcript = str(example["sentence"]).strip()

            if not transcript:
                dropped += 1
                continue

            filename = f"{kept:06d}.wav"
            wav_path = audio_dir / filename

            write_wav(
                path=wav_path,
                waveform=waveform,
                sampling_rate=sampling_rate,
            )

            tsv.write(f"{filename}\t{len(waveform)}\n")
            wrd.write(f"{transcript}\n")

            kept += 1

    print(f"{split_name}: kept={kept}, dropped={dropped}")

    return kept


def build_recipe_config(
    source_config: Path,
    output_config: Path,
    manifest_dir: Path,
):
    """Create the fairseq2 recipe configuration."""

    with source_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "dataset" not in config:
        raise ValueError(
            "The Omnilingual configuration must contain a 'dataset' section."
        )

    config["dataset"]["storage_mode"] = "MANIFEST"
    config["dataset"]["task_mode"] = "ASR"
    config["dataset"]["manifest_storage_config"] = {
        "read_text": True,
    }
    config["dataset"]["config_overrides"] = {"data": str(manifest_dir.resolve())}

    output_config.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def launch_training(
    config_path: Path,
    output_dir: Path,
):
    """Launch the official Omnilingual fairseq2 ASR recipe."""

    if not OMNILINGUAL_ROOT.exists():
        raise FileNotFoundError(
            f"Omnilingual repository not found: {OMNILINGUAL_ROOT}"
        )

    env = os.environ.copy()
    env["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"

    command = [
        sys.executable,
        "-m",
        "workflows.recipes.wav2vec2.asr",
        str(output_dir.resolve()),
        "--config-file",
        str(config_path.resolve()),
    ]

    print("Launching official Omnilingual Wav2Vec2 ASR recipe:")
    print(" ".join(command))

    subprocess.run(
        command,
        cwd=OMNILINGUAL_ROOT,
        env=env,
        check=True,
    )


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    work_dir = Path(args.work_dir).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Luhya benchmark dataset...")

    datasets = load_asr_dataset(
        dataset_name=args.dataset_name,
        train_split="train",
        eval_split="validation",
        test_split="test",
        sample=True,
        sample_size=3600,
        validation_split_pct=0.2,
        seed=42,
    )

    print(
        "Dataset sizes:",
        {
            "train": len(datasets["train"]),
            "validation": len(datasets["validation"]),
            "test": len(datasets["test"]),
        },
    )

    # The test split is intentionally NEVER materialized.
    train_count = prepare_split(
        dataset=datasets["train"],
        split_name="train",
        output_dir=work_dir,
        sampling_rate=args.sampling_rate,
        max_audio_length=args.max_audio_length,
    )

    valid_count = prepare_split(
        dataset=datasets["validation"],
        split_name="valid",
        output_dir=work_dir,
        sampling_rate=args.sampling_rate,
        max_audio_length=args.max_audio_length,
    )

    print(
        f"Materialized dataset: "
        f"train={train_count}, "
        f"validation={valid_count}"
    )

    manifest_dir = work_dir / "manifest"
    generated_config = work_dir / "omnilingual_recipe.yaml"

    build_recipe_config(
        source_config=config_path,
        output_config=generated_config,
        manifest_dir=manifest_dir,
    )

    print(f"Generated recipe config: {generated_config}")
    print(f"Manifest directory: {manifest_dir}")

    if args.dry_run:
        print("Dry run requested; training was not launched.")
        return

    launch_training(
        config_path=generated_config,
        output_dir=work_dir / "training",
    )


if __name__ == "__main__":
    main()

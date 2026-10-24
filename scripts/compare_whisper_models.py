#!/usr/bin/env python3
"""Evaluate zero-shot and fine-tuned Whisper models on Luhya ASR."""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path(
    "/network/scratch/a/ahomagnr/low-resource-asr-benchmark"
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from african_asr.data.loaders import load_asr_dataset
from african_asr.models.whisper import load_whisper


MODEL_IDS = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}

FINETUNED_MODELS = {
    "small": (
        "outputs/whisper-small_luhya_642h_50h_seed42/final_model"
    ),
    "medium": (
        "outputs/whisper-medium-luhya_642h_50h_seed42/checkpoint-5600"
    ),
    "large-v3": (
        "outputs/whisper-large-v3-luhya_642h_50h_seed42/checkpoint-1600"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot and fine-tuned Whisper models."
    )
    parser.add_argument("--target-train-hours", type=float, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum test examples; 0 evaluates the complete test split.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(evaluator_path, predictions_path, output_dir):
    subprocess.run(
        [
            sys.executable,
            str(evaluator_path),
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"Metrics file was not produced: {metrics_path}")

    return load_json(metrics_path)


def evaluate_model(model, processor, device, test_dataset, model_name):
    model.eval()
    model.generation_config.forced_decoder_ids = None
    model.generation_config.task = "transcribe"

    records = []

    for index, example in enumerate(test_dataset):
        audio = example["audio"]

        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        )

        features = inputs.input_features.to(
            device=device,
            dtype=torch.float16,
        )

        with torch.inference_mode():
            generated = model.generate(
                features,
                max_new_tokens=225,
            )

        prediction = processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

        records.append({
            "reference": str(example["sentence"]),
            "prediction": prediction,
        })

        if (index + 1) % 50 == 0:
            print(
                f"{model_name}: {index + 1}/{len(test_dataset)}",
                flush=True,
            )

    return records


def main():
    args = parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Run inside a GPU allocation."
        )

    reports_dir = PROJECT_ROOT / "reports"
    zero_shot_root = reports_dir / "zero_shot"
    fine_tuned_root = reports_dir / "fine_tuned"
    fine_tuned_root.mkdir(parents=True, exist_ok=True)

    evaluator_path = (
        PROJECT_ROOT / "src" / "african_asr" / "evaluation"
        / "compute_asr_metrics.py"
    )
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"Metric utility not found: {evaluator_path}")

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    for name, relative_path in FINETUNED_MODELS.items():
        checkpoint = SCRATCH_ROOT / relative_path
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"Fine-tuned checkpoint for {name} not found: {checkpoint}"
            )

    dataset = load_asr_dataset(
        dataset_name="Digital-Divide-Data/Luhya-ASR-Data-subset-642H",
        train_split="train",
        text_column="transcripts",
        audio_column="audio",
        sample=True,
        target_train_hours=args.target_train_hours,
        validation_split_pct=0.1,
        test_split_pct=0.1,
        seed=args.seed,
    )

    test_dataset = dataset["test"]
    if args.max_samples > 0:
        test_dataset = test_dataset.select(
            range(min(args.max_samples, len(test_dataset)))
        )

    print(f"Test examples: {len(test_dataset)}", flush=True)
    rows = []

    for model_name, model_id in MODEL_IDS.items():
        output_dir = zero_shot_root / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.json"

        print(f"\nZero-shot evaluation: {model_id}", flush=True)

        model = processor = None
        try:
            model, processor, device = load_whisper(
                model_name=model_id,
                device="cuda",
                torch_dtype="float16",
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

            records = evaluate_model(
                model, processor, device, test_dataset,
                f"zero-shot {model_name}",
            )

            with predictions_path.open("w", encoding="utf-8") as file:
                json.dump(records, file, ensure_ascii=False, indent=2)

        finally:
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            torch.cuda.empty_cache()

        metrics = calculate_metrics(
            evaluator_path, predictions_path, output_dir
        )
        rows.append({
            "model": f"whisper-{model_name}",
            "setting": "zero-shot",
            "checkpoint": model_id,
            "num_examples": len(records),
            "wer": metrics.get("wer"),
            "cer": metrics.get("cer"),
        })

    for model_name, relative_path in FINETUNED_MODELS.items():
        checkpoint = SCRATCH_ROOT / relative_path
        output_dir = fine_tuned_root / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.json"

        print(f"\nFine-tuned evaluation: {checkpoint}", flush=True)

        model = processor = None
        try:
            processor = AutoProcessor.from_pretrained(
                str(checkpoint), local_files_only=True
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(checkpoint),
                torch_dtype=torch.float16,
                local_files_only=True,
                low_cpu_mem_usage=True,
            ).to("cuda")

            records = evaluate_model(
                model, processor, "cuda", test_dataset,
                f"fine-tuned {model_name}",
            )

            with predictions_path.open("w", encoding="utf-8") as file:
                json.dump(records, file, ensure_ascii=False, indent=2)

        finally:
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            torch.cuda.empty_cache()

        metrics = calculate_metrics(
            evaluator_path, predictions_path, output_dir
        )
        rows.append({
            "model": f"whisper-{model_name}",
            "setting": "fine-tuned",
            "checkpoint": str(checkpoint),
            "num_examples": len(records),
            "wer": metrics.get("wer"),
            "cer": metrics.get("cer"),
        })

    columns = [
        "model", "setting", "checkpoint",
        "num_examples", "wer", "cer",
    ]

    csv_path = reports_dir / "model_comparison.csv"
    json_path = reports_dir / "model_comparison.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = __import__("csv").DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    print("\nMetric table saved:", flush=True)
    print(csv_path, flush=True)
    print(json_path, flush=True)


if __name__ == "__main__":
    main()

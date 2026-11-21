"""Common evaluation utilities for the African ASR benchmark."""

import argparse
import json
from pathlib import Path

import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR predictions using a common WER/CER protocol."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a JSON file containing predictions and references.",
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation",
    )
    return parser.parse_args()


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(references, predictions):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    wer = wer_metric.compute(
        references=references,
        predictions=predictions,
    )

    cer = cer_metric.compute(
        references=references,
        predictions=predictions,
    )

    return {
        "wer": wer,
        "cer": cer,
        "num_examples": len(references),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_predictions(args.predictions)

    references = [record["reference"] for record in records]
    predictions = [record["prediction"] for record in records]

    metrics = compute_metrics(
        references=references,
        predictions=predictions,
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Number of examples: {metrics['num_examples']}")
    print(f"WER: {metrics['wer']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")


if __name__ == "__main__":
    main()

"""Wav2Vec2-BERT CTC fine-tuning using the shared HF Trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from transformers import (
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
)

from src.data.loaders import load_dataset_with_splits
from src.data.preprocessing import (
    normalize_transcription,
    prepare_ctc_dataset,
)
from src.data.collator import DataCollatorCTCWithPadding
from src.models.w2v_bert import load_w2v_bert_ctc
from src.training.trainer import (
    set_seed,
    create_ctc_trainer,
    train_ctc_model,
)


DEFAULT_CONFIG = "configs/w2v_bert.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Explicit Hugging Face Trainer checkpoint path.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_vocabulary(dataset, text_column, lowercase=True):
    """Build the CTC vocabulary using training transcripts only."""

    characters = set()

    for text in dataset[text_column]:
        cleaned = normalize_transcription(
            text,
            lowercase=lowercase,
            strip_diacritics=False,
        )
        characters.update(cleaned)

    # Hugging Face CTC tokenizer uses | for word boundaries.
    characters.discard(" ")
    characters.discard("|")

    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "|": 2,
    }

    for char in sorted(characters):
        if char not in vocabulary:
            vocabulary[char] = len(vocabulary)

    return vocabulary


def create_processor(model_name, vocabulary, output_dir):
    """Create the Wav2Vec2-BERT feature extractor and CTC tokenizer."""

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    vocab_file = tokenizer_dir / "vocab.json"

    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(
            vocabulary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        do_lower_case=False,
    )

    extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        model_name
    )

    processor = Wav2Vec2BertProcessor(
        feature_extractor=extractor,
        tokenizer=tokenizer,
    )

    processor.save_pretrained(str(output_dir / "processor"))

    return processor


def build_optimizer(model, config):
    """Keep distinct learning rates for encoder and CTC head."""

    encoder_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("wav2vec2_bert."):
            encoder_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    parameter_groups = []

    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": float(config["encoder_learning_rate"]),
            }
        )

    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": float(config["learning_rate"]),
            }
        )

    if not parameter_groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(
        parameter_groups,
        betas=(
            float(config.get("adam_beta1", 0.9)),
            float(config.get("adam_beta2", 0.98)),
        ),
        eps=float(config.get("adam_epsilon", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    set_seed(int(config.get("seed", 42)))

    output_dir = Path(config["output_dir"]).expanduser()

    # Never overwrite a previous custom-DDP experiment.
    legacy_dir = Path(
        "/network/scratch/a/ahomagnr/"
        "low-resource-asr-benchmark/outputs/"
        "w2v-bert-gusii-ddp-2gpu"
    )

    if output_dir.resolve() == legacy_dir.resolve():
        raise RuntimeError(
            "Refusing to overwrite legacy custom-DDP checkpoints."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)

    # This pipeline always selects the lowest validation WER.
    config["best_model_metric"] = "wer"
    config["best_model_mode"] = "min"

    # Use official validation split only.
    dataset = load_dataset_with_splits(
        config["dataset_name"],
        audio_column=config.get("audio_column", "audio"),
        sampling_rate=16000,
    )

    if "train" not in dataset:
        raise RuntimeError("Official train split is missing.")

    if "validation" not in dataset:
        raise RuntimeError(
            "Official validation/dev split is missing. "
            "No automatic split will be created."
        )

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    if config.get("max_train_samples") is not None:
        n = min(
            int(config["max_train_samples"]),
            len(train_dataset),
        )
        train_dataset = train_dataset.select(range(n))

    if config.get("max_eval_samples") is not None:
        n = min(
            int(config["max_eval_samples"]),
            len(eval_dataset),
        )
        eval_dataset = eval_dataset.select(range(n))

    text_column = config.get("text_column", "transcripts")
    audio_column = config.get("audio_column", "audio")

    vocabulary = build_vocabulary(
        train_dataset,
        text_column=text_column,
    )

    blank_id = int(config.get("blank_index", 0))

    if vocabulary["[PAD]"] != blank_id:
        raise ValueError(
            "CTC blank index must match tokenizer pad_token_id."
        )

    processor = create_processor(
        config["model_name"],
        vocabulary,
        output_dir,
    )

    train_encoded = prepare_ctc_dataset(
        train_dataset,
        processor,
        text_column=text_column,
        audio_column=audio_column,
        max_audio_length=config.get("max_audio_length", 30.0),
        lowercase=True,
        strip_diacritics=False,
        desc="Encode Wav2Vec2-BERT train",
    )

    eval_encoded = prepare_ctc_dataset(
        eval_dataset,
        processor,
        text_column=text_column,
        audio_column=audio_column,
        max_audio_length=config.get("max_audio_length", 30.0),
        lowercase=True,
        strip_diacritics=False,
        desc="Encode Wav2Vec2-BERT validation",
    )

    if len(train_encoded) == 0 or len(eval_encoded) == 0:
        raise RuntimeError("Empty train or validation dataset.")

    print(
        f"Train={len(train_encoded)} "
        f"Validation={len(eval_encoded)} "
        f"Vocab={len(vocabulary)}",
        flush=True,
    )

    model, _ = load_w2v_bert_ctc(
        model_name=config["model_name"],
        vocab_size=len(processor.tokenizer),
        blank_token_id=processor.tokenizer.pad_token_id,
        device="cpu",
    )

    if config.get("freeze_wav2vec", False):
        for parameter in model.wav2vec2_bert.parameters():
            parameter.requires_grad = False

    collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True,
    )

    # Pilot: validate actual features and finite CTC loss.
    pilot_batch = collator([train_encoded[0]])

    model.eval()

    with torch.no_grad():
        pilot_output = model(**pilot_batch)

    if (
        pilot_output.loss is None
        or not torch.isfinite(pilot_output.loss)
    ):
        raise RuntimeError("Pilot CTC loss is not finite.")

    print(
        "Pilot CTC OK:",
        tuple(pilot_batch["input_features"].shape),
        "loss:",
        float(pilot_output.loss),
        flush=True,
    )

    model.train()

    trainer = create_ctc_trainer(
        model=model,
        processor=processor,
        train_dataset=train_encoded,
        eval_dataset=eval_encoded,
        data_collator=collator,
        config=config,
        output_dir=str(output_dir),
    )

    # Override the default Trainer optimizer to retain
    # separate encoder/head learning rates.
    trainer.optimizer = build_optimizer(
        trainer.model,
        config,
    )

    results = train_ctc_model(
        trainer,
        resume_from_checkpoint=args.resume_from_checkpoint,
        evaluate_after_training=True,
    )

    if trainer.is_world_process_zero():
        with open(
            output_dir / "config.yaml",
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(
                config,
                f,
                allow_unicode=True,
            )

        summary = {
            "best_checkpoint": results["best_checkpoint"],
            "best_validation_wer": results["best_validation_wer"],
            "validation_metrics": results["validation_metrics"],
        }

        with open(
            output_dir / "summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

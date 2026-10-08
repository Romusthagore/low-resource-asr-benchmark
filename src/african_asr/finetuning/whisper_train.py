"""
Fine-tune Whisper on an ASR dataset.

This implementation follows the validated Whisper reference training
pipeline used for the Luhya ASR experiment.

Usage with YAML config:
    python -m african_asr.finetuning.whisper_train \
        --config configs/whisper_luhya.yaml

Usage with command line:
    python -m african_asr.finetuning.whisper_train \
        --language luhya \
        --output_dir ./outputs/whisper-small-luhya
"""

import argparse
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
import yaml
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.whisper.tokenization_whisper import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
)

from african_asr.data.loaders import load_asr_dataset
from african_asr.models.whisper import load_whisper
from african_asr.utils.compute import ComputeTracker


def load_config_and_merge(args, cli_supplied: set) -> argparse.Namespace:
    if not args.config:
        return args

    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    mapping = {
        "dataset_name": "dataset_name",
        "dataset_config": "dataset_config",
        "train_split": "train_split",
        "eval_split": "eval_split",
        "text_column": "text_column",
        "audio_column": "audio_column",
        "sample": "sample",
        "sample_size": "sample_size",
        "num_proc": "num_proc",
        "validation_split_pct": "validation_split_pct",
        "test_split_pct": "test_split_pct",
        "seed": "seed",
        "model_name": "model_name",
        "language": "language",
        "task": "task",
        "torch_dtype": "torch_dtype",
        "low_cpu_mem_usage": "low_cpu_mem_usage",
        "use_safetensors": "use_safetensors",
        "output_dir": "output_dir",
        "per_device_train_batch_size": "per_device_train_batch_size",
        "per_device_eval_batch_size": "per_device_eval_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "learning_rate": "learning_rate",
        "warmup_steps": "warmup_steps",
        "num_train_epochs": "num_train_epochs",
        "max_steps": "max_steps",
        "eval_steps": "eval_steps",
        "save_steps": "save_steps",
        "logging_steps": "logging_steps",
        "save_total_limit": "save_total_limit",
        "early_stopping_patience": "early_stopping_patience",
        "group_by_length": "group_by_length",
        "gradient_checkpointing": "gradient_checkpointing",
        "fp16": "fp16",
        "max_audio_length": "max_audio_length",
        "push_to_hub": "push_to_hub",
        "hub_model_id": "hub_model_id",
    }

    for yaml_key, arg_key in mapping.items():
        if yaml_key in config and arg_key not in cli_supplied:
            setattr(args, arg_key, config[yaml_key])

    print(f"Loaded config from: {args.config}")
    return args


def add_arguments(p):
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. Command line args override YAML.",
    )

    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--eval_csv", type=str, default=None)

    p.add_argument(
        "--dataset_name",
        type=str,
        default="DDD-Kenya/Luhya-ASR-Data-subset-50h",
    )

    p.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset config/subset, if the dataset has one.",
    )

    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--text_column", type=str, default="transcript")
    p.add_argument("--audio_column", type=str, default="audio")

    p.add_argument("--sample", action="store_true", default=True)

    p.add_argument(
        "--no_sample",
        dest="sample",
        action="store_false",
        help="Disable sampling and use the full dataset split.",
    )

    p.add_argument(
    "--sample_size",
    type=int,
    default=None,
    help="Number of training examples to sample. Set in the YAML config.",
)
    p.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes used for dataset preprocessing. Set in the YAML config.",
    )
    p.add_argument("--validation_split_pct", type=float, default=0.2)
    p.add_argument("--test_split_pct", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small",
    )

    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="Target language. Required either here or via --config.",
    )

    p.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
    )

    p.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        choices=["float16", "float32"],
    )

    p.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--no_low_cpu_mem_usage",
        dest="low_cpu_mem_usage",
        action="store_false",
    )

    p.add_argument(
        "--use_safetensors",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--no_use_safetensors",
        dest="use_safetensors",
        action="store_false",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default="./whisper-small-finetuned",
    )

    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)

    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )

    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--num_train_epochs", type=float, default=2)

    p.add_argument(
        "--max_steps",
        type=int,
        default=-1,
    )

    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
    )

    p.add_argument(
        "--group_by_length",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--fp16",
        action="store_true",
        default=torch.cuda.is_available(),
    )

    p.add_argument(
        "--max_audio_length",
        type=float,
        default=30.0,
    )

    p.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
    )

    p.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
    )


def parse_args():
    p = argparse.ArgumentParser()
    add_arguments(p)

    args = p.parse_args()

    import sys

    cli_supplied = set()

    for action in p._actions:
        if not action.option_strings:
            continue

        if any(opt in sys.argv for opt in action.option_strings):
            cli_supplied.add(action.dest)

    args = load_config_and_merge(
        args,
        cli_supplied,
    )

    if not args.language:
        p.error(
            "--language is required "
            "(pass it on the command line or set it in --config's YAML file)."
        )

    return args


def resolve_whisper_language(language: str):
    lang = language.lower()

    if lang in LANGUAGES or lang in TO_LANGUAGE_CODE:
        return lang

    warnings.warn(
        f"'{language}' is not one of Whisper's supported languages. "
        "Proceeding WITHOUT a forced language token -- Whisper will rely "
        "purely on the fine-tuning data to learn the language. Flag this "
        "as a difference in starting conditions when comparing against a "
        "baseline model that natively supports more languages.",
        stacklevel=2,
    )

    return None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self,
        features: List[
            Dict[str, Union[List[int], torch.Tensor]]
        ],
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {"input_features": f["input_features"]}
            for f in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [
            {"input_ids": f["labels"]}
            for f in features
        ]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if (
            labels[:, 0]
            == self.processor.tokenizer.bos_token_id
        ).all().cpu().item():

            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main():
    args = parse_args()

    # --------------------------------------------------------------
    # Whisper language
    # --------------------------------------------------------------

    whisper_language = resolve_whisper_language(
        args.language
    )

    # --------------------------------------------------------------
    # Whisper model and processor
    # --------------------------------------------------------------

    model, processor, device = load_whisper(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        use_safetensors=args.use_safetensors,
    )

    processor_kwargs = {
        "task": args.task,
    }

    if whisper_language is not None:
        processor_kwargs["language"] = whisper_language

    # The model loader provides the processor. Re-create it with the
    # same language/task settings used by the validated reference pipeline.
    processor = type(processor).from_pretrained(
        args.model_name,
        **processor_kwargs,
    )

    if whisper_language is not None:
        model.generation_config.language = whisper_language
        model.generation_config.task = args.task
        model.generation_config.forced_decoder_ids = None

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    dataset = load_asr_dataset(
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        text_column=args.text_column,
        audio_column=args.audio_column,
        sample=args.sample,
        sample_size=args.sample_size,
        validation_split_pct=args.validation_split_pct,
        test_split_pct=args.test_split_pct,
        seed=args.seed,
    )

    # --------------------------------------------------------------
    # Whisper preprocessing
    # --------------------------------------------------------------

    def prepare_example(batch):
        audio = batch["audio"]

        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        batch["labels"] = processor.tokenizer(
            batch["sentence"]
        ).input_ids

        batch["audio_duration"] = (
            len(audio["array"])
            / audio["sampling_rate"]
        )

        return batch

    dataset = dataset.map(
        prepare_example,
        remove_columns=[
            c
            for c in dataset["train"].column_names
            if c not in ("audio_duration",)
        ],
        num_proc=args.num_proc,
    )

    dataset = dataset.filter(
        lambda x: x["audio_duration"]
        <= args.max_audio_length
    )

    dataset = dataset.remove_columns(
        ["audio_duration"]
    )

    print(
        f"Train examples: {len(dataset['train'])} | "
        f"Eval examples: {len(dataset['validation'])}"
    )

    # --------------------------------------------------------------
    # Data collator
    # --------------------------------------------------------------

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor
    )

    # --------------------------------------------------------------
    # WER
    # --------------------------------------------------------------

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[
            label_ids == -100
        ] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )

        label_str = processor.tokenizer.batch_decode(
            label_ids,
            skip_special_tokens=True,
        )

        wer = (
            100
            * wer_metric.compute(
                predictions=pred_str,
                references=label_str,
            )
        )

        return {"wer": wer}

    # --------------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------------

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=(
            args.per_device_train_batch_size
        ),
        per_device_eval_batch_size=(
            args.per_device_eval_batch_size
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        group_by_length=args.group_by_length,
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
    )

    if args.max_steps and args.max_steps > 0:
        training_args_kwargs["max_steps"] = (
            args.max_steps
        )
    else:
        training_args_kwargs["num_train_epochs"] = (
            args.num_train_epochs
        )

    accepted = set(
        inspect.signature(
            Seq2SeqTrainingArguments.__init__
        ).parameters
    )

    dropped = [
        k
        for k in training_args_kwargs
        if k not in accepted
    ]

    if dropped:
        warnings.warn(
            "Seq2SeqTrainingArguments in your installed "
            "transformers version doesn't accept: "
            f"{dropped}. Dropping them and continuing.",
            stacklevel=2,
        )

        training_args_kwargs = {
            k: v
            for k, v in training_args_kwargs.items()
            if k in accepted
        }

    training_args = Seq2SeqTrainingArguments(
        **training_args_kwargs
    )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    trainer_accepted = set(
        inspect.signature(
            Seq2SeqTrainer.__init__
        ).parameters
    )

    if (
        "processing_class" not in trainer_accepted
        and "tokenizer" in trainer_accepted
    ):
        trainer_kwargs["tokenizer"] = (
            trainer_kwargs.pop("processing_class")
        )

    if args.early_stopping_patience > 0:
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(
                early_stopping_patience=(
                    args.early_stopping_patience
                )
            )
        ]

        print(
            "Early stopping enabled with "
            f"patience={args.early_stopping_patience}"
        )

    trainer = Seq2SeqTrainer(
        **trainer_kwargs
    )

    # --------------------------------------------------------------
    # Train + compute measurement
    # --------------------------------------------------------------

    compute_tracker = ComputeTracker()
    compute_tracker.start()

    trainer.train()

    compute_results = compute_tracker.stop()

    compute_tracker.save(
        compute_results,
        f"{args.output_dir}/compute.json",
    )

    print("Compute results:")

    for key, value in compute_results.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------------
    # Save model
    # --------------------------------------------------------------

    trainer.save_model(
        args.output_dir
    )

    processor.save_pretrained(
        args.output_dir
    )

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()

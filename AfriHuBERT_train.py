"""
Fine-tune ajesujoba/AfriHuBERT (a HuBERT-family self-supervised speech
encoder for African languages) on a small ASR dataset via CTC, configured
to match the same comparison conditions as train.py (Whisper) and the
w2v-BERT-2.0 baseline: same dataset, same ~15h volume, same epochs.

AfriHuBERT ships as a bare encoder (no CTC head), so this script builds a
character-level vocabulary from the training transcripts and attaches a
fresh CTC head on top -- the same pattern your w2v-BERT-2.0 baseline used
(you'll see 'lm_head'/adapter weights reported as MISSING at load time;
that's expected, not an error).

Usage:
    python train_afrihubert.py --config configs/config_afrihubert.yaml
    python train_afrihubert.py --language luhya --output_dir /content/afrihubert-luhya
"""

import argparse
import inspect
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import evaluate
import jiwer
import torch
import yaml
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    HubertForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)


# --------------------------------------------------------------------------
# Args + YAML config merge (same pattern as train.py)
# --------------------------------------------------------------------------

def add_arguments(p):
    p.add_argument("--config", type=str, default=None,
                    help="Path to YAML config file. Command line args override YAML.")
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--eval_csv", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default="DDD-Kenya/Luhya-ASR-Data-subset-50h")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--text_column", type=str, default="transcript")
    p.add_argument("--audio_column", type=str, default="audio")

    p.add_argument("--sample", action="store_true", default=True)
    p.add_argument("--no_sample", dest="sample", action="store_false")
    p.add_argument("--sample_size", type=int, default=3600)
    p.add_argument("--validation_split_pct", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model_name", type=str, default="ajesujoba/AfriHuBERT")
    p.add_argument("--language", type=str, default=None,
                    help="Target language label, used only for logging/output naming here "
                         "(CTC models have no language-conditioning token).")
    p.add_argument("--freeze_feature_encoder", action="store_true", default=True)
    p.add_argument("--character_set", type=str,
                    default=" !'*,-.0123456789;:?abcdefghijklmnopqrstuvwxyz",
                    help="Fallback character set if a training example contains a "
                         "character outside what's found in the data (rare). The actual "
                         "vocab is built from the training transcripts themselves.")

    p.add_argument("--output_dir", type=str, default="/content/afrihubert-finetuned")
    p.add_argument("--sync_to_drive", type=str, default=None,
                    help="If set, copy the final saved model/processor here after training. "
                         "Keep --output_dir on local disk during training -- see train.py's "
                         "notes on Drive I/O reliability.")

    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--num_train_epochs", type=float, default=2)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--group_by_length", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    p.add_argument("--max_audio_length", type=float, default=30.0)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--push_to_hub", action="store_true", default=False)
    p.add_argument("--hub_model_id", type=str, default=None)


def load_config_and_merge(args, cli_supplied: set) -> argparse.Namespace:
    if not args.config:
        return args
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    mapping = {k: k for k in vars(args) if k != "config"}
    for yaml_key, arg_key in mapping.items():
        if yaml_key in config and arg_key not in cli_supplied:
            setattr(args, arg_key, config[yaml_key])

    print(f"Loaded config from: {args.config}")
    return args


def parse_args():
    p = argparse.ArgumentParser()
    add_arguments(p)
    args = p.parse_args()

    cli_supplied = set()
    for action in p._actions:
        if not action.option_strings:
            continue
        if any(opt in sys.argv for opt in action.option_strings):
            cli_supplied.add(action.dest)

    args = load_config_and_merge(args, cli_supplied)
    return args


# --------------------------------------------------------------------------
# Data loading (same sampling/splitting logic as train.py)
# --------------------------------------------------------------------------

def load_data(args) -> DatasetDict:
    if args.train_csv:
        data_files = {"train": args.train_csv}
        if args.eval_csv:
            data_files["validation"] = args.eval_csv
        ds = load_dataset("csv", data_files=data_files)
        if args.audio_column != "audio":
            ds = ds.rename_column(args.audio_column, "audio")
        ds = ds.rename_column(args.text_column, "sentence")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        if "validation" not in ds:
            split = ds["train"].train_test_split(test_size=args.validation_split_pct, seed=args.seed)
            ds = DatasetDict(train=split["train"], validation=split["test"])
        return ds

    if args.dataset_name:
        ds_all = load_dataset(args.dataset_name, args.dataset_config)

        if args.eval_split in ds_all:
            train_raw = ds_all[args.train_split]
            eval_raw = ds_all[args.eval_split]
        else:
            split = ds_all[args.train_split].train_test_split(
                test_size=args.validation_split_pct, seed=args.seed
            )
            train_raw, eval_raw = split["train"], split["test"]

        if args.sample:
            train_raw = train_raw.shuffle(seed=args.seed).select(
                range(min(args.sample_size, len(train_raw)))
            )
            eval_n = max(1, int(args.sample_size * args.validation_split_pct))
            eval_raw = eval_raw.shuffle(seed=args.seed).select(
                range(min(eval_n, len(eval_raw)))
            )

        ds = DatasetDict(train=train_raw, validation=eval_raw)
        if args.audio_column != "audio":
            ds = ds.rename_column(args.audio_column, "audio")
        if args.text_column != "sentence":
            ds = ds.rename_column(args.text_column, "sentence")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds

    raise ValueError("Provide either --train_csv (+ optional --eval_csv) or --dataset_name.")


def build_vocab(dataset, extra_chars: str) -> Dict[str, int]:
    """Build a character-level CTC vocabulary from the training transcripts,
    same approach your w2v-BERT-2.0 baseline used."""
    chars = set(extra_chars.lower())
    for text in dataset["sentence"]:
        chars.update(text.lower())
    chars.discard(" ")
    vocab_list = sorted(chars)
    vocab = {c: i for i, c in enumerate(vocab_list)}
    vocab["|"] = len(vocab)  # word delimiter (space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    print(f"Size of vocabulary created: {len(vocab)}.")
    return vocab


# --------------------------------------------------------------------------
# Data collator for CTC (pads raw audio input_values and label ids separately)
# --------------------------------------------------------------------------

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()

    dataset = load_data(args)

    def prepare_audio_duration(batch):
        audio = batch["audio"]
        batch["audio_duration"] = len(audio["array"]) / audio["sampling_rate"]
        return batch

    dataset = dataset.map(prepare_audio_duration, num_proc=1)
    dataset = dataset.filter(lambda x: x["audio_duration"] <= args.max_audio_length)
    dataset = dataset.remove_columns(["audio_duration"])

    vocab = build_vocab(dataset["train"], args.character_set)
    vocab_dir = os.path.join(args.output_dir, "ctc_tokenizer")
    os.makedirs(vocab_dir, exist_ok=True)
    vocab_path = os.path.join(vocab_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Vocabulary saved to {vocab_dir}")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = HubertForCTC.from_pretrained(
        args.model_name,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def prepare_example(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["labels"] = processor.tokenizer(batch["sentence"].lower()).input_ids
        return batch

    dataset = dataset.map(
        prepare_example,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    print(f"Train examples: {len(dataset['train'])} | Eval examples: {len(dataset['validation'])}")

    data_collator = DataCollatorCTCWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions_dir = os.path.join(args.output_dir, "predictions_json")
    os.makedirs(predictions_dir, exist_ok=True)
    eval_call_counter = {"n": 0}

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = pred_logits.argmax(axis=-1) if hasattr(pred_logits, "argmax") else pred_logits

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        score = 100 - (wer * 100 + cer * 100) / 2

        eval_call_counter["n"] += 1
        call_idx = eval_call_counter["n"]
        predictions_path = os.path.join(predictions_dir, f"predictions_{call_idx}.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump({"predictions": pred_str, "references": label_str}, f, ensure_ascii=False, indent=2)
        print(f"predictions and references saved to {predictions_path}")

        sample_indices = random.sample(range(len(pred_str)), min(10, len(pred_str)))
        for i, idx in enumerate(sample_indices):
            p_, r_ = pred_str[idx], label_str[idx]
            if r_.strip():
                sample_wer = jiwer.wer(r_, p_) * 100
                sample_cer = jiwer.cer(r_, p_) * 100
            else:
                sample_wer = sample_cer = 0.0
            print(f"Sample {i}:")
            print(f"Prediction: {p_}")
            print(f"Reference: {r_}")
            print(f"WER: {sample_wer:.4f}%")
            print(f"CER: {sample_cer:.4f}%")
            print("-" * 75)

        return {"wer": wer, "cer": cer, "score": score}

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        group_by_length=args.group_by_length,
        eval_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
    )
    if args.max_steps and args.max_steps > 0:
        training_args_kwargs["max_steps"] = args.max_steps
    else:
        training_args_kwargs["num_train_epochs"] = args.num_train_epochs

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = [k for k in training_args_kwargs if k not in accepted]
    if dropped:
        warnings.warn(
            f"TrainingArguments in your installed transformers version doesn't accept: "
            f"{dropped}. Dropping them and continuing.",
            stacklevel=2,
        )
        training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if k in accepted}

    training_args = TrainingArguments(**training_args_kwargs)

    from transformers import EarlyStoppingCallback
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    trainer_accepted = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" not in trainer_accepted and "tokenizer" in trainer_accepted:
        trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
    if args.early_stopping_patience > 0:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        print(f"Early stopping enabled with patience={args.early_stopping_patience}")

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.sync_to_drive:
        import shutil
        print(f"Copying final model from {args.output_dir} to {args.sync_to_drive} ...")
        shutil.copytree(args.output_dir, args.sync_to_drive, dirs_exist_ok=True)
        print("Copy complete.")

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()

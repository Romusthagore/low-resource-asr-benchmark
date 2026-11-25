"""Monolingual MMS-1B adapter + CTC fine-tuning with DDP."""

import argparse
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from safetensors.torch import save_file, load_file
from datasets import Audio, load_dataset, load_from_disk
from jiwer import cer, wer
from transformers import (
    EarlyStoppingCallback,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


def normalize(text):
    text = unicodedata.normalize("NFC", str(text).lower())
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_split(name, split, config_name=None):
    path = Path(name).expanduser()
    if path.exists():
        try:
            obj = load_from_disk(str(path))
            return obj[split] if hasattr(obj, "keys") else obj
        except (FileNotFoundError, ValueError):
            if config_name is None:
                return load_dataset(str(path), split=split)
            return load_dataset(str(path), config_name, split=split)

    if config_name is None:
        return load_dataset(name, split=split)

    return load_dataset(name, config_name, split=split)


def build_processor(train_ds, text_column, output_dir):
    texts = [normalize(t) for t in train_ds[text_column]]
    chars = sorted(set("".join(texts)) - {" "})

    vocab = {"<pad>": 0, "<unk>": 1, "|": 2}
    for char in chars:
        if char not in vocab:
            vocab[char] = len(vocab)

    processor_dir = output_dir / "processor"
    processor_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = processor_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )

    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(str(processor_dir))
    return processor


@dataclass
class CTCDataCollator:
    processor: Wav2Vec2Processor

    def __call__(self, features):
        inputs = [
            {"input_values": f["input_values"]}
            for f in features
        ]
        labels = [
            {"input_ids": f["labels"]}
            for f in features
        ]

        batch = self.processor.pad(
            inputs, padding=True, return_tensors="pt"
        )
        label_batch = self.processor.tokenizer.pad(
            labels, padding=True, return_tensors="pt"
        )
        batch["labels"] = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        return batch


def initialize_mms(cfg, processor):
    model_name = cfg.get("model_name", "facebook/mms-1b-all")
    source_lang = cfg.get("source_adapter_lang", "swh")

    # Load the original MMS adapter using its native vocabulary.
    # Do not resize the CTC head before loading the source adapter.
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        target_lang=source_lang,
        ignore_mismatched_sizes=True,
    )

    # from_pretrained(target_lang=...) normally loads this adapter.
    # Load explicitly only if it was not activated.
    if model.target_lang != source_lang:
        model.load_adapter(source_lang)

    # The target language has its own CTC vocabulary.
    old_head = model.lm_head
    target_vocab_size = len(processor.tokenizer)

    model.lm_head = nn.Linear(
        old_head.in_features,
        target_vocab_size,
        bias=old_head.bias is not None,
    )

    model.config.vocab_size = target_vocab_size
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = True

    # Freeze the complete model.
    for param in model.parameters():
        param.requires_grad = False

    # Native MMS adapter includes attention adapters AND lm_head.
    adapter_weights = model._get_adapters()

    for param in adapter_weights.values():
        param.requires_grad = True

    adapter_params = sum(
        p.numel()
        for name, p in adapter_weights.items()
        if not name.startswith("lm_head.")
    )
    head_params = sum(
        p.numel()
        for name, p in adapter_weights.items()
        if name.startswith("lm_head.")
    )
    total_params = sum(p.numel() for p in model.parameters())

    trainable_names = {
        name for name, p in model.named_parameters()
        if p.requires_grad
    }

    assert trainable_names == set(adapter_weights), (
        "Trainable parameters differ from native MMS adapters"
    )
    assert adapter_params > 0, "No trainable MMS adapters found"
    assert head_params > 0, "CTC head is frozen"

    print(f"Source adapter: {source_lang}", flush=True)
    print(f"Target vocabulary: {target_vocab_size}", flush=True)
    print(f"Trainable adapters: {adapter_params:,}", flush=True)
    print(f"Trainable CTC head: {head_params:,}", flush=True)
    print(
        f"Trainable total: {adapter_params + head_params:,}"
        f" / {total_params:,}",
        flush=True,
    )

    return model



def save_and_verify_adapter(model, processor, best_dir, lang):
    """Save native MMS adapter weights and verify exact reload."""
    best_dir.mkdir(parents=True, exist_ok=True)

    adapter_state = {
        name: param.detach().cpu().contiguous().clone()
        for name, param in model._get_adapters().items()
    }

    filename = best_dir / f"adapter.{lang}.safetensors"

    save_file(
        adapter_state,
        str(filename),
        metadata={"format": "pt"},
    )

    restored = load_file(str(filename), device="cpu")

    if set(restored) != set(adapter_state):
        raise RuntimeError("Saved adapter keys do not match")

    for name, original in adapter_state.items():
        if not torch.equal(original, restored[name]):
            raise RuntimeError(
                f"Adapter round-trip mismatch: {name}"
            )

    processor.save_pretrained(str(best_dir))

    with open(
        best_dir / "adapter_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "language": lang,
                "format": "native_mms_adapter",
                "vocab_size": len(processor.tokenizer),
                "tensors": len(adapter_state),
                "parameters": sum(
                    t.numel() for t in adapter_state.values()
                ),
            },
            f,
            indent=2,
        )

    print(
        f"Adapter saved and verified: {filename}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-config", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rank = int(os.environ.get("RANK", "0"))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["dataset_name"]
    dataset_config = cfg.get("dataset_config")
    if args.dataset_config is not None:
        dataset_config = args.dataset_config
    audio_column = cfg.get("audio_column", "audio")
    text_column = cfg.get("text_column", "transcripts")

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if distributed:
        if rank == 0:
            train_ds = load_split(
                dataset_name, cfg.get("train_split", "train"), dataset_config
            )
            valid_ds = load_split(
                dataset_name, cfg.get("eval_split", "validation"), dataset_config
            )
            dist.barrier()
        else:
            dist.barrier()
            train_ds = load_split(
                dataset_name, cfg.get("train_split", "train"), dataset_config
            )
            valid_ds = load_split(
                dataset_name, cfg.get("eval_split", "validation"), dataset_config
            )
    else:
        train_ds = load_split(
            dataset_name, cfg.get("train_split", "train"), dataset_config
        )
        valid_ds = load_split(
            dataset_name, cfg.get("eval_split", "validation"), dataset_config
        )

    train_ds = train_ds.cast_column(
        audio_column, Audio(sampling_rate=16000)
    )
    valid_ds = valid_ds.cast_column(
        audio_column, Audio(sampling_rate=16000)
    )

    max_train = cfg.get("max_train_samples")
    max_eval = cfg.get("max_eval_samples")

    if max_train is not None:
        train_ds = train_ds.select(
            range(min(max_train, len(train_ds)))
        )
    if max_eval is not None:
        valid_ds = valid_ds.select(
            range(min(max_eval, len(valid_ds)))
        )

    if rank == 0:
        processor = build_processor(
            train_ds, text_column, output_dir
        )

    if distributed:
        # torchrun sets RANK/WORLD_SIZE, but Trainer has not
        # necessarily initialized the process group yet.
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        dist.barrier()

    processor = Wav2Vec2Processor.from_pretrained(
        str(output_dir / "processor")
    )

    print(
        f"Rank={rank} | Train={len(train_ds)} | "
        f"Validation={len(valid_ds)}",
        flush=True,
    )

    model = initialize_mms(cfg, processor)

    max_audio_length = cfg.get("max_audio_length", 30.0)

    def prepare(row):
        audio = row[audio_column]
        duration = len(audio["array"]) / audio["sampling_rate"]

        if duration > max_audio_length:
            row["keep"] = False
            row["input_values"] = np.zeros(0, dtype=np.float32)
            row["labels"] = []
            return row

        text = normalize(row[text_column])
        if not text:
            row["keep"] = False
            row["input_values"] = np.zeros(0, dtype=np.float32)
            row["labels"] = []
            return row

        row["input_values"] = np.asarray(
            processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
            ).input_values[0],
            dtype=np.float32,
        )

        row["labels"] = processor.tokenizer(text).input_ids
        row["keep"] = True
        return row

    def preprocess(ds):
        ds = ds.map(
            prepare,
            num_proc=cfg.get("num_proc", 2),
            desc="Preparing audio and CTC labels",
        )
        ds = ds.filter(lambda row: row["keep"])
        return ds.remove_columns(
            [c for c in ds.column_names
             if c not in ("input_values", "labels")]
        )

    train_ds = preprocess(train_ds)
    valid_ds = preprocess(valid_ds)

    if not train_ds or not valid_ds:
        raise RuntimeError("Empty train or validation dataset")

    print(
        f"Rank={rank} | Prepared train={len(train_ds)} "
        f"| validation={len(valid_ds)}",
        flush=True,
    )

    def compute_metrics(prediction):
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        pred_ids = np.argmax(logits, axis=-1)
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = (
            processor.tokenizer.pad_token_id
        )

        hypotheses = [
            normalize(x)
            for x in processor.batch_decode(pred_ids)
        ]
        references = [
            normalize(x)
            for x in processor.batch_decode(
                label_ids, group_tokens=False
            )
        ]

        return {
            "wer": wer(references, hypotheses),
            "cer": cer(references, hypotheses),
        }

    eval_steps = cfg.get("eval_steps", 200)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=cfg.get("batch_size", 2),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get(
            "gradient_accumulation", 8
        ),
        learning_rate=cfg.get("learning_rate", 1e-4),
        num_train_epochs=cfg.get("epochs", 10),
        warmup_ratio=0.08,
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=cfg.get("logging_steps", 25),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
        ddp_find_unused_parameters=True,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=CTCDataCollator(processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001,
            )
        ],
    )

    trainer.train(
        resume_from_checkpoint=cfg.get(
            "resume_from_checkpoint"
        )
    )

    metrics = trainer.evaluate()

    if trainer.is_world_process_zero():
        best_dir = output_dir / "best"
        trainer.save_model(str(best_dir))

        save_and_verify_adapter(
            trainer.model,
            processor,
            best_dir,
            cfg.get("target_adapter_lang", "guz"),
        )

        with open(
            output_dir / "metrics.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "validation": metrics,
                    "best_checkpoint":
                        trainer.state.best_model_checkpoint,
                    "source_adapter":
                        cfg.get("source_adapter_lang", "swh"),
                    "target_lang": cfg.get("target_lang"),
                    "seed": seed,
                },
                f,
                indent=2,
            )

        print(
            f"Best checkpoint: "
            f"{trainer.state.best_model_checkpoint}",
            flush=True,
        )
        print(f"Validation metrics: {metrics}", flush=True)
        print(f"Saved model: {best_dir}", flush=True)


if __name__ == "__main__":
    main()

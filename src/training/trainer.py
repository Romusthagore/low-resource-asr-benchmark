"""Shared Hugging Face Trainer utilities for CTC ASR.

Intended for:
    - Wav2Vec2-BERT
    - AfriHuBERT (after adapting its model interface)
    - MMS

This module does not modify existing training scripts or checkpoints.

Checkpoint selection:
    Best validation WER (lower is better).
    CER is reported separately.
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from transformers import Trainer, TrainingArguments

from src.evaluation.metrics import compute_asr_metrics


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_logits_for_metrics(logits, labels):
    """Keep token predictions instead of full CTC logits.

    Reduces evaluation memory usage substantially.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    return torch.argmax(logits, dim=-1)


def build_compute_metrics(processor):
    """Build a WER/CER evaluation function for CTC models."""

    if not hasattr(processor, "tokenizer"):
        raise TypeError(
            "CTC metrics require a processor with a tokenizer."
        )

    def compute_metrics(pred):
        predictions = pred.predictions

        # With preprocess_logits_for_metrics, predictions
        # already contain token IDs.
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        predictions = np.asarray(predictions)

        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        if predictions.ndim != 2:
            raise ValueError(
                f"Unexpected prediction shape: {predictions.shape}"
            )

        label_ids = np.array(pred.label_ids, copy=True)

        # Replace ignored positions before decoding references.
        label_ids[label_ids == -100] = (
            processor.tokenizer.pad_token_id
        )

        prediction_texts = processor.batch_decode(
            predictions
        )

        reference_texts = processor.batch_decode(
            label_ids,
            group_tokens=False,
        )

        metrics = compute_asr_metrics(
            reference_texts,
            prediction_texts,
        )

        return {
            "wer": float(metrics["wer"]),
            "cer": float(metrics["cer"]),
        }

    return compute_metrics


def create_training_args(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> TrainingArguments:
    """Create TrainingArguments from a benchmark configuration.

    Supports the current YAML names and common names from the
    previous Luhya training pipeline.

    Does not use any combined WER/CER score.
    """

    resolved_output = output_dir or config.get("output_dir")

    if not resolved_output:
        raise ValueError("An output_dir is required.")

    resolved_output = str(Path(resolved_output).expanduser())

    batch_size = int(config.get("batch_size", 4))

    eval_batch_size = int(
        config.get(
            "test_batch_size",
            config.get("per_device_eval_batch_size", batch_size),
        )
    )

    grad_accumulation = int(
        config.get(
            "grad_accumulation_factor",
            config.get("gradient_accumulation_steps", 1),
        )
    )

    num_epochs = float(
        config.get(
            "number_of_epochs",
            config.get("num_epochs", 10),
        )
    )

    eval_strategy = config.get("eval_strategy", "epoch")
    save_strategy = config.get("save_strategy", eval_strategy)

    if eval_strategy == "no":
        raise ValueError(
            "Validation must be enabled to select the best WER."
        )

    if save_strategy != eval_strategy:
        raise ValueError(
            "save_strategy must equal eval_strategy when "
            "load_best_model_at_end=True."
        )

    precision = str(config.get("precision", "fp32")).lower()

    fp16 = bool(config.get("fp16", precision == "fp16"))
    bf16 = bool(config.get("bf16", precision == "bf16"))

    if fp16 and bf16:
        raise ValueError("fp16 and bf16 cannot both be enabled.")

    if config.get("best_model_metric", "wer") != "wer":
        raise ValueError(
            "Best checkpoint selection must use validation WER."
        )

    if config.get("best_model_mode", "min") != "min":
        raise ValueError(
            "Best validation WER must be minimized."
        )

    kwargs = {
        "output_dir": resolved_output,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accumulation,
        "num_train_epochs": num_epochs,
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.1)),
        "warmup_steps": int(config.get("warmup_steps", 0)),
        "lr_scheduler_type": config.get(
            "lr_scheduler_type", "linear"
        ),
        "optim": config.get("optim", "adamw_torch"),
        "adam_beta1": float(config.get("adam_beta1", 0.9)),
        "adam_beta2": float(config.get("adam_beta2", 0.98)),
        "adam_epsilon": float(config.get("adam_epsilon", 1e-8)),
        "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": bool(
            config.get("gradient_checkpointing", False)
        ),
        "dataloader_num_workers": int(
            config.get("dataloader_num_workers", 4)
        ),
        "dataloader_pin_memory": bool(
            config.get("dataloader_pin_memory", True)
        ),
        "logging_strategy": config.get(
            "logging_strategy", "steps"
        ),
        "logging_steps": int(config.get("logging_steps", 50)),
        "save_strategy": save_strategy,
        "save_steps": int(
            config.get(
                "save_steps",
                config.get("checkpoint_steps", 500),
            )
        ),
        "eval_steps": int(config.get("eval_steps", 500)),
        "save_total_limit": int(
            config.get("save_total_limit", 3)
        ),
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "report_to": config.get("report_to", "none"),
        "seed": int(config.get("seed", 42)),
        "data_seed": int(config.get("seed", 42)),
        "remove_unused_columns": False,
        "group_by_length": bool(
            config.get("group_by_length", False)
        ),
        "push_to_hub": False,
        "max_steps": int(config.get("max_steps", -1)),
    }

    # Handle differences between Transformers versions.
    accepted = set(
        inspect.signature(TrainingArguments.__init__).parameters
    )

    if "eval_strategy" in accepted:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in accepted:
        kwargs["evaluation_strategy"] = eval_strategy
    else:
        raise RuntimeError(
            "Installed Transformers version does not support "
            "evaluation strategy configuration."
        )

    unsupported = [
        key for key in kwargs if key not in accepted
    ]

    if unsupported:
        raise RuntimeError(
            "Unsupported TrainingArguments parameters: "
            + ", ".join(unsupported)
        )

    if eval_strategy == "steps":
        if kwargs["eval_steps"] <= 0:
            raise ValueError("eval_steps must be positive.")

        if kwargs["save_steps"] % kwargs["eval_steps"] != 0:
            raise ValueError(
                "save_steps must be a multiple of eval_steps "
                "when selecting the best model."
            )

    return TrainingArguments(**kwargs)


def create_ctc_trainer(
    *,
    model,
    processor,
    train_dataset,
    eval_dataset,
    data_collator,
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Trainer:
    """Create a Trainer for a compatible CTC model.

    Model requirements:
      - Accepts input_values or input_features.
      - Accepts labels.
      - Returns a loss during training.
      - Returns logits during evaluation.

    AfriHuBERT's current custom model must be adapted first.
    """

    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    if eval_dataset is None or len(eval_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    training_args = create_training_args(
        config=config,
        output_dir=output_dir,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": build_compute_metrics(processor),
        "preprocess_logits_for_metrics": (
            preprocess_logits_for_metrics
        ),
    }

    accepted = set(
        inspect.signature(Trainer.__init__).parameters
    )

    if "processing_class" in accepted:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in accepted:
        trainer_kwargs["tokenizer"] = processor

    return Trainer(**trainer_kwargs)


def train_ctc_model(
    trainer: Trainer,
    *,
    resume_from_checkpoint: Optional[str] = None,
    evaluate_after_training: bool = True,
):
    """Train and optionally evaluate the best validation checkpoint.

    An explicit checkpoint path is required to resume.
    Existing experiments are never resumed automatically.
    """

    result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    trainer.save_model()

    if trainer.is_world_process_zero():
        trainer.save_state()

    if evaluate_after_training:
        validation_metrics = trainer.evaluate()
    else:
        validation_metrics = {}

    return {
        "train_result": result,
        "validation_metrics": validation_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_wer": trainer.state.best_metric,
    }

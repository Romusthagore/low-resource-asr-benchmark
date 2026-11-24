"""AfriHuBERT fine-tuning for the African ASR benchmark."""

import argparse
from contextlib import nullcontext
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import yaml
from src.evaluation.metrics import (
    compute_asr_metrics,
    decode_ctc_predictions,
)
from torch.nn.utils.rnn import pad_sequence

from src.data.loaders import load_dataset_with_splits
from src.models.afrihubert import load_afrihubert
from src.utils.compute import ComputeTracker
from src.utils.distributed import (
    setup_distributed,
    wrap_model,
    unwrap_model,
    is_main_process,
    barrier,
    cleanup_distributed,
    distributed_epoch_indices,
    gather_rank_states,
)
from src.utils.training_checkpoint import (
    capture_rng_state,
    restore_rng_state,
    save_checkpoint,
    load_checkpoint,
)


DEFAULT_MODEL = "ajesujoba/AfriHuBERT"
DEFAULT_CONFIG = "configs/afrihubert_luhya.yaml"


class AfriHuBERTCTC(nn.Module):
    """Custom CTC head for AfriHuBERT.

    NOTE: this architecture (3 feed-forward blocks of 1024 units with
    LayerNorm + LeakyReLU + dropout) has not been verified against the
    original AfriHuBERT paper. Treat it as a reasonable custom CTC head,
    not as a reproduction of a published architecture, unless confirmed
    otherwise.
    """

    def __init__(self, encoder, vocab_size):
        super().__init__()

        self.encoder = encoder

        self.linear1 = nn.Linear(encoder.config.hidden_size, 1024)
        self.norm1 = nn.LayerNorm(1024)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.15)

        self.linear2 = nn.Linear(1024, 1024)
        self.norm2 = nn.LayerNorm(1024)
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.15)

        self.linear3 = nn.Linear(1024, 1024)
        self.norm3 = nn.LayerNorm(1024)
        self.act3 = nn.LeakyReLU()

        self.ctc_lin = nn.Linear(1024, vocab_size)

    def _block(self, hidden, linear, norm, activation, dropout=None):
        hidden = linear(hidden)
        # LayerNorm normalizes over the feature dimension independently
        # for each (batch, time) position, so padded time steps don't
        # contaminate the statistics the way BatchNorm1d would.
        hidden = norm(hidden)
        hidden = activation(hidden)

        if dropout is not None:
            hidden = dropout(hidden)

        return hidden

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state

        hidden = self._block(hidden, self.linear1, self.norm1, self.act1, self.drop1)
        hidden = self._block(hidden, self.linear2, self.norm2, self.act2, self.drop2)
        hidden = self._block(hidden, self.linear3, self.norm3, self.act3)

        logits = self.ctc_lin(hidden)

        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune AfriHuBERT for ASR."
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Dataset id (local path or Hugging Face Hub repo), "
            "overriding 'dataset_name' from the config file."
        ),
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "AfriHuBERT model id (local path or Hugging Face Hub repo), "
            "overriding 'model_name' from the config file."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for checkpoints and metrics, overriding "
            "'output_dir' from the config file."
        ),
    )

    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_train_eval_splits(config):
    """Load the dataset and return (train_dataset, eval_dataset).

    Uses `load_dataset_with_splits`, which already returns canonical
    keys ("train", "validation", "test") for whichever splits actually
    exist, and already resamples audio to 16 kHz.

    If no validation split exists, a slice is carved out of train. When
    a "speaker_name" column is available, the fallback split is made
    speaker-disjoint (no speaker appears in both train and validation);
    otherwise it falls back to a plain random split, with a warning,
    since a non-disjoint split can leak speaker identity between the
    two sets.
    """

    dataset = load_dataset_with_splits(
        config["dataset_name"],
        audio_column=config.get("audio_column", "audio"),
    )

    train_dataset = dataset["train"]

    if "validation" in dataset:
        return train_dataset, dataset["validation"]

    val_pct = config.get("fallback_validation_pct", 0.1)
    print(f"No validation split found; carving out {val_pct:.0%} of train.")

    if "speaker_name" in train_dataset.column_names:
        from sklearn.model_selection import GroupShuffleSplit

        speakers = train_dataset["speaker_name"]
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_pct, random_state=config["seed"]
        )
        train_idx, val_idx = next(
            splitter.split(X=range(len(speakers)), groups=speakers)
        )
        train_split = train_dataset.select(train_idx.tolist())
        eval_split = train_dataset.select(val_idx.tolist())
        print(
            f"Speaker-disjoint fallback split: {len(train_split)} train / "
            f"{len(eval_split)} validation examples."
        )
    else:
        print(
            "WARNING: no 'speaker_name' column found; falling back to a "
            "random split that is NOT speaker-disjoint. Speaker identity "
            "may leak between train and validation."
        )
        split = train_dataset.train_test_split(test_size=val_pct, seed=config["seed"])
        train_split, eval_split = split["train"], split["test"]

    return train_split, eval_split


def prepare_examples(dataset, max_audio_length, text_column):
    examples = []

    for example in dataset:
        audio = example["audio"]
        waveform = audio["array"]
        sampling_rate = audio["sampling_rate"]

        if len(waveform) == 0:
            continue

        duration = len(waveform) / sampling_rate

        if duration > max_audio_length:
            continue

        text = example[text_column]

        if not text:
            continue

        examples.append(
            {
                "audio": waveform,
                "sampling_rate": sampling_rate,
                "sentence": text,
            }
        )

    return examples


def build_character_vocabulary(examples):
    """Build a CTC-only character vocabulary from training transcripts.

    Only a blank token and an explicit unknown-character token are
    added; <bos>/<eos> are not meaningful for CTC (no autoregressive
    decoder here) and are intentionally omitted.
    """

    characters = set()

    for example in examples:
        characters.update(example["sentence"])

    vocabulary = {
        "<blank>": 0,
        "<unk>": 1,
    }

    for character in sorted(characters):
        if character not in vocabulary:
            vocabulary[character] = len(vocabulary)

    return vocabulary


def encode_text(text, vocabulary, unk_counter=None):
    """Encode text to token ids.

    Characters absent from the training vocabulary are mapped to
    <unk> rather than silently dropped, so targets are never silently
    truncated. `unk_counter`, if given, is incremented in place for
    each substitution, so the caller can report how often this happens.
    """

    unk_id = vocabulary["<unk>"]
    tokens = []

    for character in text:
        token_id = vocabulary.get(character, unk_id)

        if token_id == unk_id and character not in vocabulary and unk_counter is not None:
            unk_counter[0] += 1

        tokens.append(token_id)

    return tokens


def collate_batch(batch, vocabulary, device, unk_counter=None):
    waveforms = []
    targets = []
    target_lengths = []

    for example in batch:
        waveform = torch.tensor(
            example["audio"],
            dtype=torch.float32,
        )

        waveforms.append(waveform)

        tokens = encode_text(
            example["sentence"],
            vocabulary,
            unk_counter=unk_counter,
        )

        target = torch.tensor(
            tokens,
            dtype=torch.long,
        )

        targets.append(target)
        target_lengths.append(len(target))

    input_values = pad_sequence(
        waveforms,
        batch_first=True,
    ).to(device)

    attention_mask = torch.zeros(
        input_values.shape,
        dtype=torch.long,
        device=device,
    )

    for i, waveform in enumerate(waveforms):
        attention_mask[i, : len(waveform)] = 1

    targets = torch.cat(targets).to(device)

    target_lengths = torch.tensor(
        target_lengths,
        dtype=torch.long,
        device=device,
    )

    input_lengths = attention_mask.sum(dim=1)

    return (
        input_values,
        attention_mask,
        targets,
        input_lengths,
        target_lengths,
    )


def sanity_check_output_lengths(model, batch, vocabulary, device):
    """Pilot-level check: verify output_lengths are consistent with the
    model's actual output before running a full training loop.

    This does not prove the conv-length formula is exactly right for
    every input, but it catches gross mismatches (e.g. output_lengths
    exceeding the logits' time dimension) early, as recommended before
    trusting `_get_feat_extract_output_lengths` for a given encoder.
    """

    model.eval()

    with torch.no_grad():
        (
            input_values,
            attention_mask,
            _targets,
            input_lengths,
            _target_lengths,
        ) = collate_batch(batch, vocabulary, device)

        logits = model(input_values=input_values, attention_mask=attention_mask)

        output_lengths = unwrap_model(model).encoder._get_feat_extract_output_lengths(
            input_lengths
        ).to(device)

        max_output_length = logits.shape[1]

        if (output_lengths > max_output_length).any():
            raise RuntimeError(
                f"Sanity check failed: computed output_lengths "
                f"(max={output_lengths.max().item()}) exceed the model's "
                f"actual output time dimension ({max_output_length}). "
                f"_get_feat_extract_output_lengths may not match this "
                f"encoder's real downsampling."
            )

        print(
            f"Sanity check passed: output_lengths in "
            f"[{output_lengths.min().item()}, {output_lengths.max().item()}], "
            f"logits time dimension = {max_output_length}."
        )

    model.train()


@torch.no_grad()
def evaluate_model(
    model,
    eval_examples,
    vocabulary,
    batch_size,
    device,
    ctc_loss,
):
    """Evaluate loss, WER and CER on the validation set."""

    model.eval()

    id_to_token = {
        index: token
        for token, index in vocabulary.items()
    }

    total_loss = 0.0
    total_batches = 0

    references = []
    predictions = []

    for start in range(
        0,
        len(eval_examples),
        batch_size,
    ):
        batch = eval_examples[
            start : start + batch_size
        ]

        (
            input_values,
            attention_mask,
            targets,
            input_lengths,
            target_lengths,
        ) = collate_batch(
            batch,
            vocabulary,
            device,
        )

        logits = model(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        log_probs = F.log_softmax(
            logits,
            dim=-1,
        )

        output_lengths = unwrap_model(model).encoder._get_feat_extract_output_lengths(
            input_lengths
        ).to(device)

        loss = ctc_loss(
            log_probs.transpose(0, 1),
            targets,
            output_lengths,
            target_lengths,
        )

        total_loss += loss.item()
        total_batches += 1

        batch_predictions = decode_ctc_predictions(
            logits,
            output_lengths,
            id_to_token,
        )

        predictions.extend(batch_predictions)
        references.extend(
            example["sentence"]
            for example in batch
        )

    average_loss = total_loss / max(1, total_batches)

    metrics = compute_asr_metrics(
        references,
        predictions,
    )

    return {
        "eval_loss": average_loss,
        **metrics,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name

    if args.model_name is not None:
        config["model_name"] = args.model_name

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    context = setup_distributed()
    distributed = context["distributed"]
    rank = context["rank"]
    world_size = context["world_size"]
    device = context["device"]

    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Distributed: {distributed}")
    print(f"Rank: {rank}/{world_size}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {config['seed']}")
    print(f"Model: {config['model_name']}")

    train_split, eval_split = load_train_eval_splits(config)

    text_column = config.get("text_column", "sentence")

    max_train_samples = config.get("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
        if max_train_samples < 1:
            raise ValueError("max_train_samples must be >= 1")
        train_split = train_split.select(
            range(min(max_train_samples, len(train_split)))
        )

    train_examples = prepare_examples(
        train_split,
        config["max_audio_length"],
        text_column,
    )

    max_eval_samples = config.get("max_eval_samples")

    if max_eval_samples is not None:
        eval_split = eval_split.select(
            range(min(max_eval_samples, len(eval_split)))
        )

    eval_examples = prepare_examples(
        eval_split,
        config["max_audio_length"],
        text_column,
    )

    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")

    vocabulary = build_character_vocabulary(
        train_examples
    )

    print(f"Vocabulary size: {len(vocabulary)}")

    if is_main_process(context):
        with open(
            output_dir / "vocabulary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                vocabulary,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Also save the config itself, for traceability.
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)


    barrier(context)

    encoder, _ = load_afrihubert(
        model_name=config["model_name"],
        device=device,
        torch_dtype="float32",
    )

    model = AfriHuBERTCTC(
        encoder=encoder,
        vocab_size=len(vocabulary),
    ).to(device)

    checkpoint_path = config.get("initial_checkpoint")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint)

        print(f"Loaded initial checkpoint: {checkpoint_path}")
    else:
        print("No initial ASR checkpoint configured.")
        print(
            "Using pretrained encoder and a newly initialized ASR head."
        )

    if config["freeze_wav2vec"]:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    model_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
        and parameter.requires_grad
    ]

    encoder_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]

    model_optimizer = torch.optim.Adadelta(
        model_parameters,
        lr=config["learning_rate"],
        rho=0.95,
        eps=1e-8,
    )

    encoder_optimizer = None

    if encoder_parameters:
        encoder_optimizer = torch.optim.Adam(
            encoder_parameters,
            lr=config["encoder_learning_rate"],
        )

    ctc_loss = nn.CTCLoss(
        blank=config["blank_index"],
        zero_infinity=True,
    )



    # Full-state checkpoint for automatic resume.
    checkpoint_file = output_dir / "last_checkpoint.pt"
    checkpoint_interval = int(config["checkpoint_steps"])
    save_each_epoch = config["save_each_epoch"]
    resume_enabled = config["resume_from_checkpoint"]
    save_best_model = config["save_best_model"]
    best_metric = config["best_model_metric"]
    best_mode = config["best_model_mode"]

    if best_metric not in ("wer", "cer", "eval_loss"):
        raise ValueError(f"Unsupported best_model_metric: {best_metric}")

    if best_mode not in ("min", "max"):
        raise ValueError(f"Unsupported best_model_mode: {best_mode}")

    if not save_best_model:
        raise ValueError(
            "save_best_model must be true because the final model "
            "is loaded from best_model.pt"
        )

    if checkpoint_interval < 1:
        raise ValueError("checkpoint_steps must be >= 1")

    metrics_history = []
    best_wer = float("inf") if best_mode == "min" else float("-inf")
    optimizer_steps = 0
    start_epoch = 0
    start_batch = 0

    resume_running_loss = 0.0
    resume_batch_count = 0
    resume_unk_count = 0

    if resume_enabled and checkpoint_file.is_file():
        checkpoint = load_checkpoint(
            checkpoint_file,
            map_location="cpu",
        )

        checkpoint_version = checkpoint.get("version")

        if checkpoint_version not in (1, 2):
            raise RuntimeError(
                f"Unsupported checkpoint format: {checkpoint_version}"
            )

        if checkpoint_version == 1 and distributed:
            raise RuntimeError(
                "Cannot resume a V1 single-GPU checkpoint directly "
                "in DDP mode."
            )

        if checkpoint_version == 2:
            saved_world_size = checkpoint["world_size"]

            if saved_world_size != world_size:
                raise RuntimeError(
                    f"Checkpoint uses {saved_world_size} processes, "
                    f"but current training uses {world_size}. "
                    "Changing world size during resume is not yet supported."
                )

        if checkpoint["seed"] != config["seed"]:
            raise RuntimeError("Checkpoint seed does not match config")

        if checkpoint["batch_size"] != config["batch_size"]:
            raise RuntimeError("Checkpoint batch size does not match config")

        if checkpoint["grad_accumulation_factor"] != config["grad_accumulation_factor"]:
            raise RuntimeError("Gradient accumulation configuration changed")

        if checkpoint["dataset_name"] != config["dataset_name"]:
            raise RuntimeError("Checkpoint dataset does not match config")

        if checkpoint["model_name"] != config["model_name"]:
            raise RuntimeError("Checkpoint model does not match config")

        if checkpoint["vocabulary"] != vocabulary:
            raise RuntimeError("Checkpoint vocabulary does not match")

        if checkpoint["train_size"] != len(train_examples):
            raise RuntimeError("Training dataset size changed")

        if checkpoint["eval_size"] != len(eval_examples):
            raise RuntimeError("Validation dataset size changed")

        model.load_state_dict(checkpoint["model"])
        model_optimizer.load_state_dict(checkpoint["model_optimizer"])

        if encoder_optimizer is not None:
            if checkpoint["encoder_optimizer"] is None:
                raise RuntimeError("Missing encoder optimizer state")
            encoder_optimizer.load_state_dict(
                checkpoint["encoder_optimizer"]
            )
        elif checkpoint["encoder_optimizer"] is not None:
            raise RuntimeError("Unexpected encoder optimizer state")

        start_epoch = checkpoint["next_epoch"]
        start_batch = checkpoint["next_batch"]
        if checkpoint.get("best_metric", "wer") != best_metric:
            raise RuntimeError("Best metric changed since checkpoint")

        if checkpoint.get("best_mode", "min") != best_mode:
            raise RuntimeError("Best metric direction changed")

        best_wer = checkpoint["best_wer"]
        optimizer_steps = checkpoint["optimizer_steps"]
        metrics_history = checkpoint["metrics_history"]

        if checkpoint_version == 2:
            if "rank_counters" not in checkpoint:
                raise RuntimeError(
                    "V2 checkpoint lacks per-rank counters; "
                    "exact mid-epoch resume is not supported."
                )

            counters = checkpoint["rank_counters"][rank]
            resume_running_loss = counters["running_loss"]
            resume_batch_count = counters["batch_count"]
            resume_unk_count = counters["unk_count"]
        else:
            resume_running_loss = checkpoint["running_loss"]
            resume_batch_count = checkpoint["batch_count"]
            resume_unk_count = checkpoint["unk_count"]

        if best_wer != (
            float("inf") if best_mode == "min" else float("-inf")
        ) and not (
            output_dir / "best_model.pt"
        ).is_file():
            raise RuntimeError(
                "Best model is missing; refusing inconsistent resume"
            )

        print(
            f"Resuming at epoch {start_epoch + 1}, "
            f"batch {start_batch + 1}, "
            f"optimizer step {optimizer_steps}",
            flush=True,
        )

    # Verify output lengths using the restored model.
    if train_examples:
        pilot_batch = train_examples[: min(4, len(train_examples))]
        sanity_check_output_lengths(
            model, pilot_batch, vocabulary, device
        )

    # Wrap only after model/optimizer checkpoint restoration.
    # The underlying model retains its original state_dict keys.
    model = wrap_model(model, context)

    # Restore RNG only after model initialization and DDP wrapping.
    if config.get("resume_from_checkpoint", False) and checkpoint_file.is_file():
        if checkpoint_version == 2:
            restore_rng_state(checkpoint["rank_rng_states"][rank])
        else:
            restore_rng_state(checkpoint["rng_state"])

    def save_training_state(
        next_epoch,
        next_batch,
        running_loss,
        batch_count,
        unk_count,
    ):
        rank_rng_states = gather_rank_states(
            capture_rng_state(), context
        )

        rank_counters = gather_rank_states(
            {
                "running_loss": running_loss,
                "batch_count": batch_count,
                "unk_count": unk_count,
            },
            context,
        )

        if not is_main_process(context):
            return

        state = {
            "version": 2,
            "world_size": world_size,
            "rank_rng_states": rank_rng_states,
            "rank_counters": rank_counters,
            "model": unwrap_model(model).state_dict(),
            "model_optimizer": model_optimizer.state_dict(),
            "encoder_optimizer": (
                encoder_optimizer.state_dict()
                if encoder_optimizer is not None else None
            ),
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "running_loss": running_loss,
            "batch_count": batch_count,
            "unk_count": unk_count,
            "best_wer": best_wer,
            "best_metric": best_metric,
            "best_mode": best_mode,
            "optimizer_steps": optimizer_steps,
            "metrics_history": metrics_history,
            "seed": config["seed"],
            "batch_size": config["batch_size"],
            "grad_accumulation_factor": config["grad_accumulation_factor"],
            "dataset_name": config["dataset_name"],
            "model_name": config["model_name"],
            "vocabulary": vocabulary,
            "train_size": len(train_examples),
            "eval_size": len(eval_examples),
            "rng_state": capture_rng_state(),
        }
        save_checkpoint(checkpoint_file, state)

    compute_tracker = ComputeTracker()

    if config.get("track_compute", True) and is_main_process(context):
        compute_tracker.start()

    model_optimizer.zero_grad()

    if encoder_optimizer is not None:
        encoder_optimizer.zero_grad()

    for epoch in range(start_epoch, config["number_of_epochs"]):
        model.train()

        epoch_indices = distributed_epoch_indices(
            dataset_size=len(train_examples),
            epoch=epoch,
            seed=config["seed"],
            rank=rank,
            world_size=world_size,
        )
        epoch_examples = [train_examples[i] for i in epoch_indices]

        total_batches = (
            len(epoch_examples) + config["batch_size"] - 1
        ) // config["batch_size"]

        first_batch = start_batch if epoch == start_epoch else 0

        running_loss = (
            resume_running_loss if epoch == start_epoch else 0.0
        )
        batch_count = (
            resume_batch_count if epoch == start_epoch else 0
        )
        unk_counter = [
            resume_unk_count if epoch == start_epoch else 0
        ]

        accum_count = 0

        for batch_index in range(first_batch, total_batches):
            start = batch_index * config["batch_size"]
            batch = epoch_examples[
                start:start + config["batch_size"]
            ]

            (
                input_values,
                attention_mask,
                targets,
                input_lengths,
                target_lengths,
            ) = collate_batch(
                batch, vocabulary, device,
                unk_counter=unk_counter,
            )

            # Normalize by the actual accumulation group size.
            accumulation = config["grad_accumulation_factor"]
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(
                accumulation, total_batches - group_start
            )

            should_step = (
                (batch_index - group_start + 1) == group_size
            )

            sync_context = (
                model.no_sync()
                if distributed and not should_step
                else nullcontext()
            )

            # DDP requires no_sync to cover both forward and backward.
            with sync_context:
                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                )

                log_probs = F.log_softmax(logits, dim=-1)

                output_lengths = (
                    unwrap_model(model).encoder._get_feat_extract_output_lengths(
                        input_lengths
                    ).to(device)
                )

                raw_loss = ctc_loss(
                    log_probs.transpose(0, 1),
                    targets,
                    output_lengths,
                    target_lengths,
                )

                (raw_loss / group_size).backward()

            running_loss += raw_loss.item()
            batch_count += 1
            accum_count += 1

            if batch_index == 0 or (batch_index + 1) % 50 == 0:
                if is_main_process(context):
                    print(
                        f"Epoch {epoch + 1}/{config['number_of_epochs']} "
                        f"- batch {batch_index + 1}/{total_batches}",
                        flush=True,
                    )

            is_last_batch = batch_index + 1 == total_batches

            if accum_count == group_size or is_last_batch:
                model_optimizer.step()

                if encoder_optimizer is not None:
                    encoder_optimizer.step()

                model_optimizer.zero_grad()

                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()

                optimizer_steps += 1
                accum_count = 0

                if (
                    optimizer_steps % checkpoint_interval == 0
                    and not is_last_batch
                ):
                    save_training_state(
                        next_epoch=epoch,
                        next_batch=batch_index + 1,
                        running_loss=running_loss,
                        batch_count=batch_count,
                        unk_count=unk_counter[0],
                    )
                    barrier(context)
                    if is_main_process(context):
                        print(
                            f"Checkpoint saved at optimizer step "
                            f"{optimizer_steps}",
                            flush=True,
                        )

        # Aggregate training statistics across all processes.
        train_stats = torch.tensor(
            [running_loss, float(batch_count), float(unk_counter[0])],
            dtype=torch.float64,
            device=device,
        )

        if distributed:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)

        global_running_loss = train_stats[0].item()
        global_batch_count = int(train_stats[1].item())
        global_unk_count = int(train_stats[2].item())

        # Only rank 0 evaluates and writes epoch metrics.
        if is_main_process(context):
            average_train_loss = (
                global_running_loss / max(1, global_batch_count)
            )

            if global_unk_count > 0:
                print(
                    f"Unknown character substitutions: "
                    f"{global_unk_count}"
                )

            validation_metrics = evaluate_model(
                model=unwrap_model(model),
                eval_examples=eval_examples,
                vocabulary=vocabulary,
                batch_size=config["test_batch_size"],
                device=device,
                ctc_loss=ctc_loss,
            )

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": average_train_loss,
                "train_unk_substitutions": global_unk_count,
                **validation_metrics,
            }

            metrics_history.append(epoch_metrics)

            with open(
                output_dir / "metrics.json", "w", encoding="utf-8"
            ) as f:
                json.dump(metrics_history, f, indent=2)

            print(
                f"Epoch {epoch + 1}/{config['number_of_epochs']} "
                f"- train_loss: {average_train_loss:.4f} "
                f"- eval_loss: {validation_metrics['eval_loss']:.4f} "
                f"- WER: {validation_metrics['wer']:.4f} "
                f"- CER: {validation_metrics['cer']:.4f}",
                flush=True,
            )

            current_metric = validation_metrics[best_metric]

            improved = (
                current_metric < best_wer
                if best_mode == "min"
                else current_metric > best_wer
            )

            if save_best_model and improved:
                best_wer = current_metric

                save_checkpoint(
                    output_dir / "best_model.pt",
                    unwrap_model(model).state_dict(),
                )

                with open(
                    output_dir / "best_metrics.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(epoch_metrics, f, indent=2)

                print(
                    f"New best model saved ({best_metric}: {best_wer:.4f})",
                    flush=True,
                )


        # Broadcast the best validation metric from rank 0.
        if distributed:
            best_score = torch.tensor(
                [best_wer],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(best_score, src=0)
            best_wer = best_score.item()

        barrier(context)

        # The next epoch starts at batch zero.
        if save_each_epoch or resume_enabled:
            save_training_state(
                next_epoch=epoch + 1,
                next_batch=0,
                running_loss=0.0,
                batch_count=0,
                unk_count=0,
            )

            print(
                f"Epoch checkpoint saved: {checkpoint_file}",
                flush=True,
            )

        barrier(context)
        start_batch = 0

    # Final artifacts are written only by rank 0.
    barrier(context)

    if is_main_process(context):
        if config.get("track_compute", True):
            compute_results = compute_tracker.stop()
            compute_tracker.save(
                compute_results,
                output_dir / "compute.json",
            )
            print("Compute:")
            print(json.dumps(compute_results, indent=2))

        # Use the best validation checkpoint as the final model.
        best_checkpoint = output_dir / "best_model.pt"

        if not best_checkpoint.is_file():
            raise RuntimeError(
                "No best checkpoint was saved during training."
            )

        best_state_dict = torch.load(
            best_checkpoint,
            map_location=device,
            weights_only=True,
        )

        unwrap_model(model).load_state_dict(best_state_dict)

        save_checkpoint(
            output_dir / "model.pt",
            unwrap_model(model).state_dict(),
        )

        print(f"Best validation {best_metric}: {best_wer:.4f}")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Final model: {output_dir / 'model.pt'}")
        print(f"Optimizer steps: {optimizer_steps}")
        print("Training complete.")

    barrier(context)
    cleanup_distributed(context)



if __name__ == "__main__":
    main()

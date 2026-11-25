"""Wav2Vec2-BERT 2.0 CTC fine-tuning with DDP."""

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from transformers import SeamlessM4TFeatureExtractor

from src.models.w2v_bert import load_w2v_bert_ctc
from src.finetuning.afrihubert_train import (
    load_train_eval_splits,
    prepare_examples,
    build_character_vocabulary,
    encode_text,
)
from src.evaluation.metrics import (
    compute_asr_metrics,
    decode_ctc_predictions,
)
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


def collate_batch(batch, vocabulary, extractor, device):
    audio = [ex["audio"] for ex in batch]

    rates = {ex["sampling_rate"] for ex in batch}
    if rates != {16000}:
        raise ValueError(f"Expected 16 kHz audio, got {rates}")

    features = extractor(
        audio,
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    if features["input_features"].shape[-1] != 160:
        raise RuntimeError("Expected 160 acoustic features")

    labels = [
        encode_text(ex["sentence"], vocabulary)
        for ex in batch
    ]

    max_length = max(map(len, labels))
    padded_labels = torch.full(
        (len(labels), max_length),
        -100,
        dtype=torch.long,
    )

    for i, tokens in enumerate(labels):
        padded_labels[i, :len(tokens)] = torch.tensor(
            tokens, dtype=torch.long
        )

    return {
        "input_features": features["input_features"].to(device),
        "attention_mask": features["attention_mask"].to(device),
        "labels": padded_labels.to(device),
    }


@torch.no_grad()
def evaluate(model, examples, vocabulary, extractor, batch_size, device):
    model.eval()

    id_to_token = {v: k for k, v in vocabulary.items()}
    predictions = []
    references = []
    losses = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        inputs = collate_batch(batch, vocabulary, extractor, device)

        outputs = model(**inputs)

        if not torch.isfinite(outputs.loss):
            raise RuntimeError("Non-finite validation CTC loss")

        lengths = model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(dim=-1)
        ).long()

        if (lengths > outputs.logits.shape[1]).any():
            raise RuntimeError("Invalid CTC output lengths")

        decoded = decode_ctc_predictions(
            outputs.logits,
            lengths,
            id_to_token,
        )

        predictions.extend(decoded)
        references.extend(ex["sentence"] for ex in batch)
        losses.append(outputs.loss.item())

    metrics = compute_asr_metrics(references, predictions)
    metrics["eval_loss"] = sum(losses) / max(1, len(losses))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/ddp/w2v_bert_gusii_2gpu.yaml",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    context = setup_distributed()
    rank = context["rank"]
    world_size = context["world_size"]
    distributed = context["distributed"]
    device = context["device"]

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process(context):
        train_split, eval_split = load_train_eval_splits(config)
        barrier(context)
    else:
        barrier(context)
        train_split, eval_split = load_train_eval_splits(config)

    if config.get("max_train_samples") is not None:
        n = min(int(config["max_train_samples"]), len(train_split))
        train_split = train_split.select(range(n))

    if config.get("max_eval_samples") is not None:
        n = min(int(config["max_eval_samples"]), len(eval_split))
        eval_split = eval_split.select(range(n))

    train_examples = prepare_examples(
        train_split,
        config["max_audio_length"],
        config["text_column"],
    )
    eval_examples = prepare_examples(
        eval_split,
        config["max_audio_length"],
        config["text_column"],
    )

    if not train_examples or not eval_examples:
        raise RuntimeError("Empty train or validation dataset")

    vocabulary = build_character_vocabulary(train_examples)

    if is_main_process(context):
        print(
            f"Train={len(train_examples)}, "
            f"Eval={len(eval_examples)}, "
            f"Vocab={len(vocabulary)}",
            flush=True,
        )
        (output_dir / "vocabulary.json").write_text(
            json.dumps(vocabulary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(config),
            encoding="utf-8",
        )

    barrier(context)

    extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        config["model_name"]
    )

    model, _ = load_w2v_bert_ctc(
        model_name=config["model_name"],
        vocab_size=len(vocabulary),
        blank_token_id=config["blank_index"],
        device=device,
    )

    if config.get("freeze_wav2vec", False):
        for parameter in model.wav2vec2_bert.parameters():
            parameter.requires_grad = False

    encoder_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("wav2vec2_bert."):
            encoder_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(config["encoder_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(config["learning_rate"]),
            },
        ]
    )

    checkpoint_file = output_dir / "last_checkpoint.pt"
    best_file = output_dir / "best_model.pt"

    start_epoch = 0
    start_batch = 0
    optimizer_steps = 0
    best_wer = float("inf")
    history = []
    saved_rng = None
    running_loss_resume = 0.0
    batch_count_resume = 0

    if config.get("resume_from_checkpoint", True) and checkpoint_file.exists():
        checkpoint = load_checkpoint(
            checkpoint_file, map_location="cpu"
        )

        if checkpoint["world_size"] != world_size:
            raise RuntimeError("Checkpoint world size mismatch")
        if checkpoint["vocabulary"] != vocabulary:
            raise RuntimeError("Checkpoint vocabulary mismatch")

        for key in (
            "dataset_name",
            "model_name",
            "seed",
            "batch_size",
            "grad_accumulation_factor",
        ):
            if checkpoint["config"][key] != config[key]:
                raise RuntimeError(f"Checkpoint config mismatch: {key}")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["next_epoch"]
        start_batch = checkpoint["next_batch"]
        optimizer_steps = checkpoint["optimizer_steps"]
        best_wer = checkpoint["best_wer"]
        history = checkpoint["history"]
        saved_rng = checkpoint["rank_rng_states"][rank]

        counters = checkpoint["rank_counters"][rank]
        running_loss_resume = counters["running_loss"]
        batch_count_resume = counters["batch_count"]

        if best_wer < float("inf") and not best_file.exists():
            raise RuntimeError("Best checkpoint missing")

        if is_main_process(context):
            print(
                f"Resuming epoch={start_epoch + 1}, "
                f"batch={start_batch + 1}",
                flush=True,
            )

    # Pilot: check real acoustic features and CTC loss.
    pilot = collate_batch(
        train_examples[:1], vocabulary, extractor, device
    )

    model.eval()
    with torch.no_grad():
        pilot_output = model(**pilot)

    if not torch.isfinite(pilot_output.loss):
        raise RuntimeError("Pilot CTC loss is not finite")

    if is_main_process(context):
        print(
            f"Pilot CTC OK: "
            f"features={tuple(pilot['input_features'].shape)}, "
            f"logits={tuple(pilot_output.logits.shape)}, "
            f"loss={pilot_output.loss.item():.4f}",
            flush=True,
        )

    del pilot, pilot_output

    model = wrap_model(model, context)

    if saved_rng is not None:
        restore_rng_state(saved_rng)

    def save_state(next_epoch, next_batch, running_loss, batch_count):
        rng_states = gather_rank_states(
            capture_rng_state(), context
        )
        counters = gather_rank_states(
            {
                "running_loss": running_loss,
                "batch_count": batch_count,
            },
            context,
        )

        if is_main_process(context):
            save_checkpoint(
                checkpoint_file,
                {
                    "version": 1,
                    "world_size": world_size,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "next_epoch": next_epoch,
                    "next_batch": next_batch,
                    "optimizer_steps": optimizer_steps,
                    "best_wer": best_wer,
                    "history": history,
                    "vocabulary": vocabulary,
                    "config": config,
                    "rank_rng_states": rng_states,
                    "rank_counters": counters,
                },
            )

    accumulation = int(config["grad_accumulation_factor"])
    batch_size = int(config["batch_size"])
    checkpoint_steps = int(config["checkpoint_steps"])

    for epoch in range(start_epoch, int(config["number_of_epochs"])):
        model.train()

        indices = distributed_epoch_indices(
            dataset_size=len(train_examples),
            epoch=epoch,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
        epoch_examples = [train_examples[i] for i in indices]
        total_batches = (len(epoch_examples) + batch_size - 1) // batch_size

        first_batch = start_batch if epoch == start_epoch else 0
        running_loss = (
            running_loss_resume if epoch == start_epoch else 0.0
        )
        batch_count = (
            batch_count_resume if epoch == start_epoch else 0
        )

        optimizer.zero_grad()

        for batch_index in range(first_batch, total_batches):
            start = batch_index * batch_size
            batch = epoch_examples[start:start + batch_size]

            inputs = collate_batch(
                batch, vocabulary, extractor, device
            )

            group_start = (batch_index // accumulation) * accumulation
            group_size = min(
                accumulation, total_batches - group_start
            )
            should_step = (
                batch_index - group_start + 1 == group_size
            )

            sync_context = (
                model.no_sync()
                if distributed and not should_step
                else nullcontext()
            )

            with sync_context:
                outputs = model(**inputs)
                loss = outputs.loss

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite CTC loss at epoch={epoch}, "
                        f"batch={batch_index}"
                    )

                (loss / group_size).backward()

            running_loss += loss.item()
            batch_count += 1

            if should_step:
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1

                if (
                    optimizer_steps % checkpoint_steps == 0
                    and batch_index + 1 < total_batches
                ):
                    save_state(
                        epoch,
                        batch_index + 1,
                        running_loss,
                        batch_count,
                    )
                    barrier(context)

            if (
                is_main_process(context)
                and (batch_index == first_batch or
                     (batch_index + 1) % 50 == 0)
            ):
                print(
                    f"Epoch {epoch + 1} "
                    f"batch {batch_index + 1}/{total_batches} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        stats = torch.tensor(
            [running_loss, float(batch_count)],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        if is_main_process(context):
            metrics = evaluate(
                unwrap_model(model),
                eval_examples,
                vocabulary,
                extractor,
                int(config["test_batch_size"]),
                device,
            )

            metrics["epoch"] = epoch + 1
            metrics["train_loss"] = (
                stats[0].item() / max(1, stats[1].item())
            )
            history.append(metrics)

            (output_dir / "metrics.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={metrics['train_loss']:.4f}, "
                f"eval_loss={metrics['eval_loss']:.4f}, "
                f"WER={metrics['wer']:.4f}, "
                f"CER={metrics['cer']:.4f}",
                flush=True,
            )

            if metrics["wer"] < best_wer:
                best_wer = metrics["wer"]
                save_checkpoint(
                    best_file,
                    unwrap_model(model).state_dict(),
                )
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(metrics, indent=2),
                    encoding="utf-8",
                )
                print("New best checkpoint saved", flush=True)

        if distributed:
            best_tensor = torch.tensor(
                [best_wer], dtype=torch.float64, device=device
            )
            dist.broadcast(best_tensor, src=0)
            best_wer = best_tensor.item()

        barrier(context)
        save_state(epoch + 1, 0, 0.0, 0)
        barrier(context)
        start_batch = 0

    if is_main_process(context):
        if not best_file.exists():
            raise RuntimeError("No best checkpoint found")

        best_state = torch.load(
            best_file, map_location=device, weights_only=True
        )
        unwrap_model(model).load_state_dict(best_state)

        save_checkpoint(
            output_dir / "model.pt",
            unwrap_model(model).state_dict(),
        )
        print(
            f"Training complete. Best WER={best_wer:.4f}",
            flush=True,
        )

    barrier(context)
    cleanup_distributed(context)


if __name__ == "__main__":
    main()

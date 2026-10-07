"""AfriHuBERT fine-tuning for the African ASR benchmark."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from jiwer import wer, cer
from torch.nn.utils.rnn import pad_sequence

from african_asr.data.loaders import load_asr_dataset
from african_asr.models.afrihubert import load_afrihubert
from african_asr.utils.compute import ComputeTracker


DEFAULT_MODEL = "ajesujoba/AfriHuBERT"
DEFAULT_CONFIG = "configs/afrihubert_luhya.yaml"


class AfriHuBERTCTC(nn.Module):
    """Official AfriHuBERT downstream ASR architecture."""

    def __init__(self, encoder, vocab_size):
        super().__init__()

        self.encoder = encoder

        self.linear1 = nn.Linear(768, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.15)

        self.linear2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.15)

        self.linear3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.act3 = nn.LeakyReLU()

        self.ctc_lin = nn.Linear(1024, vocab_size)

    def _block(self, hidden, linear, batch_norm, activation, dropout=None):
        hidden = linear(hidden)
        hidden = hidden.transpose(1, 2)
        hidden = batch_norm(hidden)
        hidden = hidden.transpose(1, 2)
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

        hidden = self._block(
            hidden,
            self.linear1,
            self.bn1,
            self.act1,
            self.drop1,
        )

        hidden = self._block(
            hidden,
            self.linear2,
            self.bn2,
            self.act2,
            self.drop2,
        )

        hidden = self._block(
            hidden,
            self.linear3,
            self.bn3,
            self.act3,
        )

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


def prepare_examples(dataset, max_audio_length):
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

        examples.append(
            {
                "audio": waveform,
                "sampling_rate": sampling_rate,
                "sentence": example["sentence"],
            }
        )

    return examples


def build_character_vocabulary(examples):
    """Build character vocabulary from training transcripts."""

    characters = set()

    for example in examples:
        characters.update(example["sentence"])

    vocabulary = {
        "<blank>": 0,
        "<bos>": 1,
        "<eos>": 2,
    }

    for character in sorted(characters):
        if character not in vocabulary:
            vocabulary[character] = len(vocabulary)

    return vocabulary


def encode_text(text, vocabulary):
    return [
        vocabulary[character]
        for character in text
        if character in vocabulary
    ]


def collate_batch(batch, vocabulary, device):
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


def decode_ctc_predictions(logits, output_lengths, id_to_token):
    """Greedy CTC decoding."""

    predictions = logits.argmax(dim=-1)

    decoded = []

    for prediction, length in zip(predictions, output_lengths):
        sequence = prediction[: int(length)].tolist()

        tokens = []
        previous = None

        for token_id in sequence:
            if token_id == 0:
                previous = token_id
                continue

            if token_id == previous:
                continue

            tokens.append(id_to_token[token_id])
            previous = token_id

        decoded.append("".join(tokens))

    return decoded


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

        output_lengths = model.encoder._get_feat_extract_output_lengths(
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

    validation_wer = wer(
        references,
        predictions,
    )

    validation_cer = cer(
        references,
        predictions,
    )

    return {
        "eval_loss": average_loss,
        "wer": validation_wer,
        "cer": validation_cer,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {config['seed']}")
    print(f"Model: {config['model_name']}")

    dataset = load_asr_dataset(
        dataset_name=config["dataset_name"],
        dataset_config=config.get("dataset_config"),
        train_split=config["train_split"],
        eval_split=config["eval_split"],
        text_column=config["text_column"],
        audio_column=config["audio_column"],
        sample=config["sample"],
        sample_size=config["sample_size"],
        validation_split_pct=config["validation_split_pct"],
        seed=config["seed"],
    )

    train_examples = prepare_examples(
        dataset["train"],
        config["max_audio_length"],
    )

    eval_examples = prepare_examples(
        dataset["validation"],
        config["max_audio_length"],
    )

    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")

    vocabulary = build_character_vocabulary(
        train_examples
    )

    print(f"Vocabulary size: {len(vocabulary)}")

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

    encoder, _ = load_afrihubert(
        model_name=config["model_name"],
        device=device,
        torch_dtype="float32",
    )

    model = AfriHuBERTCTC(
        encoder=encoder,
        vocab_size=len(vocabulary),
    ).to(device)

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

    compute_tracker = ComputeTracker()

    if config.get("track_compute", True):
        compute_tracker.start()

    model_optimizer.zero_grad()

    if encoder_optimizer is not None:
        encoder_optimizer.zero_grad()

    metrics_history = []

    best_wer = float("inf")

    global_step = 0
    optimizer_steps = 0

    for epoch in range(
        config["number_of_epochs"]
    ):
        model.train()

        running_loss = 0.0
        batch_count = 0

        total_batches = (
            len(train_examples)
            + config["batch_size"]
            - 1
        ) // config["batch_size"]

        for batch_index, start in enumerate(
            range(
                0,
                len(train_examples),
                config["batch_size"],
            ),
            start=1,
        ):
            batch = train_examples[
                start : start + config["batch_size"]
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

            output_lengths = (
                model.encoder
                ._get_feat_extract_output_lengths(
                    input_lengths
                )
                .to(device)
            )

            raw_loss = ctc_loss(
                log_probs.transpose(0, 1),
                targets,
                output_lengths,
                target_lengths,
            )

            running_loss += raw_loss.item()
            batch_count += 1

            loss = (
                raw_loss
                / config["grad_accumulation_factor"]
            )

            loss.backward()

            global_step += 1

            is_last_batch = (
                batch_index == total_batches
            )

            should_step = (
                global_step
                % config["grad_accumulation_factor"]
                == 0
            ) or is_last_batch

            if should_step:
                model_optimizer.step()

                if encoder_optimizer is not None:
                    encoder_optimizer.step()

                model_optimizer.zero_grad()

                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()

                optimizer_steps += 1

        average_train_loss = (
            running_loss / max(1, batch_count)
        )

        validation_metrics = evaluate_model(
            model=model,
            eval_examples=eval_examples,
            vocabulary=vocabulary,
            batch_size=config["test_batch_size"],
            device=device,
            ctc_loss=ctc_loss,
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": average_train_loss,
            **validation_metrics,
        }

        metrics_history.append(epoch_metrics)

        with open(
            output_dir / "metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metrics_history,
                f,
                indent=2,
            )

        print(
            f"Epoch {epoch + 1}/{config['number_of_epochs']} "
            f"- train_loss: {average_train_loss:.4f} "
            f"- eval_loss: {validation_metrics['eval_loss']:.4f} "
            f"- WER: {validation_metrics['wer']:.4f} "
            f"- CER: {validation_metrics['cer']:.4f}"
        )

        if validation_metrics["wer"] < best_wer:
            best_wer = validation_metrics["wer"]

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt",
            )

            with open(
                output_dir / "best_metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    epoch_metrics,
                    f,
                    indent=2,
                )

            print(
                f"New best model saved "
                f"(WER: {best_wer:.4f})"
            )

    if config.get("track_compute", True):
        compute_results = compute_tracker.stop()

        compute_tracker.save(
            compute_results,
            output_dir / "compute.json",
        )

        print("Compute:")
        print(
            json.dumps(
                compute_results,
                indent=2,
            )
        )

    torch.save(
        model.state_dict(),
        output_dir / "model.pt",
    )

    print(f"Optimizer steps: {optimizer_steps}")
    print("Training complete.")


if __name__ == "__main__":
    main()

"""Shared ASR metrics and CTC decoding utilities."""

import torch
from jiwer import wer, cer


def compute_asr_metrics(references, predictions):
    """Compute corpus-level WER and CER."""

    if len(references) != len(predictions):
        raise ValueError(
            "References and predictions must have equal lengths."
        )

    if not references:
        raise ValueError("Cannot evaluate an empty dataset.")

    return {
        "wer": wer(references, predictions),
        "cer": cer(references, predictions),
    }


def decode_ctc_predictions(
    logits,
    output_lengths,
    id_to_token,
    blank_id=0,
):
    """Greedy CTC decoding."""

    predictions = logits.argmax(dim=-1)
    decoded = []

    for prediction, length in zip(predictions, output_lengths):
        sequence = prediction[:int(length)].tolist()

        tokens = []
        previous = None

        for token_id in sequence:
            if token_id != blank_id and token_id != previous:
                tokens.append(id_to_token[token_id])

            previous = token_id

        decoded.append("".join(tokens))

    return decoded

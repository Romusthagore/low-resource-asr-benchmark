"""Whisper model utilities for the African ASR benchmark."""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_WHISPER_MODEL = "openai/whisper-small"


def load_whisper(
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str | None = None,
):
    """Load a Whisper model and its processor."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    model.to(device)

    return model, processor, device

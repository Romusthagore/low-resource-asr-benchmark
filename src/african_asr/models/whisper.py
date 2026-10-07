"""Whisper model utilities for the African ASR benchmark."""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_WHISPER_MODEL = "openai/whisper-small"


def load_whisper(
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str | None = None,
    torch_dtype: str | None = None,
    low_cpu_mem_usage: bool = True,
    use_safetensors: bool = True,
):
    """Load a Whisper model and its processor."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(
            f"Unsupported torch_dtype: {torch_dtype}"
        )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_safetensors=use_safetensors,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    model.to(device)

    return model, processor, device

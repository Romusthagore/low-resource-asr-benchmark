"""BuzzASR model utilities for the African ASR benchmark."""

import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

DEFAULT_BUZZ_MODEL = "BuzzASR/swahili"


def load_buzz(
    model_name: str = DEFAULT_BUZZ_MODEL,
    device: str | None = None,
    torch_dtype: str | None = None,
    low_cpu_mem_usage: bool = True,
    use_safetensors: bool = True,
):
    """Load BuzzASR directly, without relying on the Whisper benchmark loader."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        None: torch.float32,
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    dtype = dtype_map[torch_dtype]

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_safetensors=use_safetensors,
        local_files_only=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=True,
    )

    model.to(device)

    return model, processor, device

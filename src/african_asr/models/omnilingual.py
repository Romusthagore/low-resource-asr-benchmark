"""Omnilingual ASR model utilities for the African ASR benchmark."""

import torch

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


DEFAULT_OMNILINGUAL_MODEL = "omniASR_CTC_300M_v2"


def load_omnilingual(
    model_card: str = DEFAULT_OMNILINGUAL_MODEL,
    device: str | None = None,
    dtype: str | None = None,
):
    """Load an Omnilingual ASR inference pipeline."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(
            f"Unsupported dtype: {dtype}"
        )

    pipeline = ASRInferencePipeline(
        model_card=model_card,
        device=device,
        dtype=torch_dtype,
    )

    return pipeline, device

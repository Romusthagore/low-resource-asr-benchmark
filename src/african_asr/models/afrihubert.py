"""AfriHuBERT model utilities for the African ASR benchmark."""

import torch
from transformers import HubertModel


DEFAULT_AFRIHUBERT_MODEL = "ajesujoba/AfriHuBERT"


def load_afrihubert(
    model_name: str = DEFAULT_AFRIHUBERT_MODEL,
    device: str | None = None,
    torch_dtype: str | None = None,
):
    """Load the pretrained AfriHuBERT encoder."""

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

    model = HubertModel.from_pretrained(
        model_name,
        dtype=dtype,
    )

    model.to(device)

    return model, device

"""Wav2Vec2-BERT 2.0 model utilities for the African ASR benchmark."""

import torch
from transformers import Wav2Vec2BertModel


DEFAULT_W2V_BERT_MODEL = "facebook/w2v-bert-2.0"


def load_w2v_bert(
    model_name: str = DEFAULT_W2V_BERT_MODEL,
    device: str | None = None,
    torch_dtype: str | None = None,
):
    """Load the pretrained Wav2Vec2-BERT 2.0 encoder."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    model = Wav2Vec2BertModel.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model.to(device)

    return model, device


from transformers import Wav2Vec2BertForCTC


def load_w2v_bert_ctc(
    model_name=DEFAULT_W2V_BERT_MODEL,
    vocab_size=None,
    blank_token_id=0,
    device=None,
):
    """Load Wav2Vec2-BERT 2.0 with a CTC classification head."""

    if vocab_size is None:
        raise ValueError("vocab_size must be specified")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Wav2Vec2BertForCTC.from_pretrained(
        model_name,
        vocab_size=vocab_size,
        pad_token_id=blank_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        dtype=torch.float32,
    )

    model.to(device)

    return model, device

"""Dynamic padding collator for compatible Hugging Face CTC processors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DataCollatorCTCWithPadding:
    """Pad variable-length audio inputs and mask padded CTC labels with -100.

    Compatible with processors that support `.pad(...)` for input_values or
    input_features and expose `.tokenizer.pad(...)` for CTC labels.
    """

    processor: Any
    padding: bool = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch")

        if "input_features" in features[0]:
            input_key = "input_features"
        elif "input_values" in features[0]:
            input_key = "input_values"
        else:
            raise KeyError("Expected 'input_features' or 'input_values'")

        if any(input_key not in f or "labels" not in f for f in features):
            raise ValueError(
                "Every example must have the same input key and labels"
            )

        input_features = [
            {input_key: f[input_key]}
            for f in features
        ]

        label_features = [
            {"input_ids": f["labels"]}
            for f in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        return batch

"""Collators for training and export workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import DataCollatorForTokenClassification


def get_token_classification_collator(tokenizer: Any) -> DataCollatorForTokenClassification:
    """Build the standard Hugging Face token classification collator."""
    return DataCollatorForTokenClassification(tokenizer=tokenizer)


@dataclass
class ExportCollator:
    """Pad model inputs while keeping rich metadata as Python lists."""

    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        model_keys = {"input_ids", "attention_mask", "token_type_ids"}
        batch_inputs = []
        metadata: dict[str, list[Any]] = {}

        for feature in features:
            batch_inputs.append({key: value for key, value in feature.items() if key in model_keys})
            for key, value in feature.items():
                if key in model_keys:
                    continue
                metadata.setdefault(key, []).append(value)

        batch = self.tokenizer.pad(batch_inputs, padding=True, return_tensors="pt")
        batch.update(metadata)
        return batch

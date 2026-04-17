"""Model and tokenizer loading helpers for DeBERTa token classification."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


def load_tokenizer(model_name_or_path: str) -> Any:
    """Load a fast tokenizer for token classification and vector export."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("This pipeline requires a fast tokenizer for word alignment.")
    return tokenizer


def load_token_classification_model(
    model_name_or_path: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    output_hidden_states: bool = False,
) -> Any:
    """Load an encoder-only token classification model with explicit label mappings."""
    hf_id2label = {int(key): value for key, value in id2label.items()}
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=hf_id2label,
        output_hidden_states=output_hidden_states,
    )
    return AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=config)


def load_checkpoint_model(checkpoint_path: str, output_hidden_states: bool = False) -> Any:
    """Load a fine-tuned checkpoint for inference or vector export."""
    config = AutoConfig.from_pretrained(checkpoint_path, output_hidden_states=output_hidden_states)
    return AutoModelForTokenClassification.from_pretrained(checkpoint_path, config=config)

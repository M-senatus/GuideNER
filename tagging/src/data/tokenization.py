"""Tokenizer-based preprocessing for training and vector export."""

from __future__ import annotations

from typing import Any

from datasets import Dataset

from .schemas import NERExample


def examples_to_dataset(examples: list[NERExample]) -> Dataset:
    """Convert typed examples into a Hugging Face Dataset."""
    return Dataset.from_list([example.to_record() for example in examples])


def tokenize_and_align_labels(dataset: Dataset, tokenizer: Any, label2id: dict[str, int], max_length: int) -> Dataset:
    """Tokenize word-level NER data and align labels to the first subword only."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for word alignment.")
    if "has_labels" in dataset.column_names:
        unlabeled_count = sum(1 for flag in dataset["has_labels"] if not flag)
        if unlabeled_count:
            raise ValueError(
                f"tokenize_and_align_labels received {unlabeled_count} unlabeled examples."
            )

    def _tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

        all_labels: list[list[int]] = []
        for batch_index, tags in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_idx = None
            label_ids: list[int] = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[tags[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            all_labels.append(label_ids)

        tokenized["labels"] = all_labels
        return tokenized

    return dataset.map(
        _tokenize_batch,
        batched=True,
        desc="Tokenizing NER dataset",
    )


def build_inference_features(examples: list[NERExample], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    """Prepare padded-agnostic features with word alignment metadata for inference/export."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for inference/export alignment.")

    features: list[dict[str, Any]] = []
    for example in examples:
        encoded = tokenizer(
            example.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        word_ids = [(-1 if idx is None else int(idx)) for idx in encoded.word_ids()]
        offset_mapping = [[int(start), int(end)] for start, end in encoded["offset_mapping"]]

        feature = {
            "sample_id": example.sample_id,
            "split": example.split,
            "source_path": example.source_path,
            "has_labels": example.has_labels,
            "tokens": list(example.tokens),
            "ner_tags": list(example.ner_tags),
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "word_ids": word_ids,
            "offset_mapping": offset_mapping,
        }
        if "token_type_ids" in encoded:
            feature["token_type_ids"] = list(encoded["token_type_ids"])
        features.append(feature)
    return features

"""Label helpers for BIO NER tagging."""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import NERExample, SpanAnnotation


def _label_sort_key(label: str) -> tuple[int, str, int]:
    if label == "O":
        return (0, "", 0)
    prefix, _, entity = label.partition("-")
    prefix_rank = {"B": 0, "I": 1}.get(prefix, 2)
    return (1, entity, prefix_rank)


def build_label_mappings(examples: Iterable[NERExample]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build deterministic label mappings from training examples."""
    labels = set()
    for example in examples:
        if not example.has_labels:
            raise ValueError(
                f"Cannot build label mappings from unlabeled example '{example.sample_id}'."
            )
        labels.update(example.ner_tags)
    labels = sorted(labels, key=_label_sort_key)
    if "O" not in labels:
        labels = ["O"] + labels
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


def validate_example_labels(examples: Iterable[NERExample], allowed_labels: set[str]) -> None:
    """Ensure that all labels in the examples belong to an expected label set."""
    for example in examples:
        if not example.has_labels:
            raise ValueError(
                f"Cannot validate label vocabulary for unlabeled example '{example.sample_id}'."
            )
        unknown = set(example.ner_tags) - allowed_labels
        if unknown:
            raise ValueError(
                f"Example '{example.sample_id}' contains unknown labels: {sorted(unknown)}"
            )


def extract_spans_from_bio(example: NERExample) -> list[SpanAnnotation]:
    """Extract labeled entity spans from a BIO tag sequence."""
    if not example.has_labels:
        raise ValueError(
            f"Cannot extract gold spans from unlabeled example '{example.sample_id}'."
        )
    spans: list[SpanAnnotation] = []
    start: int | None = None
    entity_type: str | None = None

    def close_span(end: int) -> None:
        nonlocal start, entity_type
        if start is None or entity_type is None:
            return
        spans.append(
            SpanAnnotation(
                sample_id=example.sample_id,
                start=start,
                end=end,
                entity_type=entity_type,
                metadata={"span_text": " ".join(example.tokens[start:end])},
            )
        )
        start = None
        entity_type = None

    for idx, tag in enumerate(example.ner_tags):
        if tag == "O":
            close_span(idx)
            continue

        prefix, _, current_type = tag.partition("-")
        if prefix == "B":
            close_span(idx)
            start = idx
            entity_type = current_type
        elif prefix == "I":
            if start is None or entity_type != current_type:
                close_span(idx)
                start = idx
                entity_type = current_type
        else:
            raise ValueError(f"Unsupported BIO tag prefix in '{tag}'.")

    close_span(len(example.ner_tags))
    return spans

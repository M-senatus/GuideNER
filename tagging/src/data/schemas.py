"""Typed schemas shared across the tagging pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NERExample:
    """A single tokenized NER sample at the word level."""

    sample_id: str
    tokens: list[str]
    ner_tags: list[str]
    split: str
    source_path: str
    has_labels: bool = True

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.ner_tags):
            raise ValueError(
                f"Token/tag length mismatch for sample '{self.sample_id}': "
                f"{len(self.tokens)} tokens vs {len(self.ner_tags)} tags."
            )

    def to_record(self) -> dict[str, Any]:
        """Convert the example into a JSON-serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "tokens": list(self.tokens),
            "ner_tags": list(self.ner_tags),
            "has_labels": self.has_labels,
            "split": self.split,
            "source_path": self.source_path,
        }


@dataclass
class SpanAnnotation:
    """A labeled or unlabeled span on word indices [start, end)."""

    sample_id: str
    start: int
    end: int
    entity_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"Invalid span for sample '{self.sample_id}': start={self.start}, end={self.end}."
            )

    def to_record(self) -> dict[str, Any]:
        """Convert the span into a JSON-serializable dictionary."""
        payload = {
            "sample_id": self.sample_id,
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
        }
        payload.update(self.metadata)
        return payload

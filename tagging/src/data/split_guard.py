"""Hard split-isolation guards for train/dev/test workflows."""

from __future__ import annotations

from pathlib import Path


ALLOWED_STAGES = frozenset({"train", "dev", "test_infer", "final_eval"})
SPLIT_ALIASES = {
    "train": "train",
    "dev": "dev",
    "validation": "dev",
    "valid": "dev",
    "test": "test",
}


class DatasetStageError(RuntimeError):
    """Raised when a script tries to access a split from the wrong stage."""


def normalize_stage(stage: str) -> str:
    """Normalize and validate a stage name."""
    normalized = str(stage).strip().lower()
    if normalized not in ALLOWED_STAGES:
        raise DatasetStageError(
            f"Unsupported stage '{stage}'. Expected one of {sorted(ALLOWED_STAGES)}."
        )
    return normalized


def normalize_split(split: str) -> str:
    """Normalize a split alias into train/dev/test."""
    normalized = str(split).strip().lower()
    if normalized not in SPLIT_ALIASES:
        raise DatasetStageError(
            f"Unsupported split '{split}'. Expected one of {sorted(SPLIT_ALIASES)}."
        )
    return SPLIT_ALIASES[normalized]


def _format_source_hint(source_path: str | None) -> str:
    if source_path is None:
        return ""
    return f" Source: {Path(source_path).resolve()}."


def ensure_train_access(stage: str, operation: str, source_path: str | None = None) -> str:
    """Allow labeled train access only during the train stage."""
    normalized_stage = normalize_stage(stage)
    if normalized_stage != "train":
        raise DatasetStageError(
            f"{operation} is only allowed in stage='train', got stage='{normalized_stage}'."
            f"{_format_source_hint(source_path)}"
        )
    return normalized_stage


def ensure_dev_access(stage: str, operation: str, source_path: str | None = None) -> str:
    """Allow labeled dev access only during the dev stage."""
    normalized_stage = normalize_stage(stage)
    if normalized_stage != "dev":
        raise DatasetStageError(
            f"{operation} is only allowed in stage='dev', got stage='{normalized_stage}'."
            f"{_format_source_hint(source_path)}"
        )
    return normalized_stage


def ensure_test_text_only_access(stage: str, operation: str, source_path: str | None = None) -> str:
    """Allow text-only test access only during test inference."""
    normalized_stage = normalize_stage(stage)
    if normalized_stage != "test_infer":
        raise DatasetStageError(
            f"{operation} is only allowed in stage='test_infer', got stage='{normalized_stage}'."
            f"{_format_source_hint(source_path)}"
        )
    return normalized_stage


def ensure_final_eval_access(stage: str, operation: str, source_path: str | None = None) -> str:
    """Allow labeled test access only during final evaluation."""
    normalized_stage = normalize_stage(stage)
    if normalized_stage != "final_eval":
        raise DatasetStageError(
            f"{operation} is only allowed in stage='final_eval', got stage='{normalized_stage}'."
            f"{_format_source_hint(source_path)}"
        )
    return normalized_stage


def placeholder_ner_tags(token_count: int) -> list[str]:
    """Return label placeholders for text-only inference examples."""
    return ["O"] * max(0, int(token_count))

"""Stage-gated loaders for the repository-level GuideNER JSONL dataset files."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tagging.src.data.split_guard import (
    ensure_dev_access,
    ensure_final_eval_access,
    ensure_test_text_only_access,
    ensure_train_access,
)


JSON_VALUE_DECODER = json.JSONDecoder()
JSON_KEY_PATTERN = r'"{key}"\s*:'


def _dataset_file(dataset_dir: str | Path, split_name: str) -> Path:
    return Path(dataset_dir).resolve() / f"{split_name}.jsonl"


def _extract_json_field(raw_line: str, key: str) -> object | None:
    match = re.search(JSON_KEY_PATTERN.format(key=re.escape(key)), raw_line)
    if match is None:
        return None
    value_start = match.end()
    while value_start < len(raw_line) and raw_line[value_start].isspace():
        value_start += 1
    value, _ = JSON_VALUE_DECODER.raw_decode(raw_line, idx=value_start)
    return value


def _load_labeled_sentence_records(file_path: Path, split_name: str) -> list[dict]:
    records: list[dict] = []
    with file_path.open("r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "text" not in record or "entity_labels" not in record:
                raise ValueError(
                    f"Labeled record at {file_path}:{idx + 1} must contain 'text' and 'entity_labels'."
                )
            if not isinstance(record["text"], str):
                raise ValueError(f"'text' must be a string at {file_path}:{idx + 1}.")
            if not isinstance(record["entity_labels"], list):
                raise ValueError(f"'entity_labels' must be a list at {file_path}:{idx + 1}.")
            records.append(
                {
                    "sample_id": str(record.get("sample_id", f"{split_name}-{idx}")),
                    "text": record["text"],
                    "entity_labels": record["entity_labels"],
                }
            )
    return records


def _load_text_only_sentence_records(file_path: Path, split_name: str) -> list[dict]:
    records: list[dict] = []
    with file_path.open("r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue

            text = _extract_json_field(stripped, "text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Text-only test record at {file_path}:{idx + 1} must expose a string 'text' field."
                )

            sample_id = _extract_json_field(stripped, "sample_id")
            records.append(
                {
                    "sample_id": str(sample_id) if sample_id is not None else f"{split_name}-{len(records)}",
                    "text": text,
                }
            )
    return records


def load_train_data(dataset_dir: str | Path, stage: str = "train") -> list[dict]:
    """Load labeled train sentence records. Only valid during stage='train'."""
    file_path = _dataset_file(dataset_dir, "train")
    ensure_train_access(stage, operation="load_train_data()", source_path=str(file_path))
    return _load_labeled_sentence_records(file_path, split_name="train")


def load_dev_data(dataset_dir: str | Path, stage: str = "dev") -> list[dict]:
    """Load labeled dev sentence records. Only valid during stage='dev'."""
    file_path = _dataset_file(dataset_dir, "dev")
    ensure_dev_access(stage, operation="load_dev_data()", source_path=str(file_path))
    return _load_labeled_sentence_records(file_path, split_name="dev")


def load_test_text_only(dataset_dir: str | Path, stage: str = "test_infer") -> list[dict]:
    """Load test text without labels. Only valid during stage='test_infer'."""
    file_path = _dataset_file(dataset_dir, "test")
    ensure_test_text_only_access(stage, operation="load_test_text_only()", source_path=str(file_path))
    return _load_text_only_sentence_records(file_path, split_name="test")


def load_test_with_labels_for_final_eval(dataset_dir: str | Path, stage: str = "final_eval") -> list[dict]:
    """Load labeled test sentence records. Only valid during stage='final_eval'."""
    file_path = _dataset_file(dataset_dir, "test")
    ensure_final_eval_access(
        stage,
        operation="load_test_with_labels_for_final_eval()",
        source_path=str(file_path),
    )
    return _load_labeled_sentence_records(file_path, split_name="test")


def load_label_schema(dataset_dir: str | Path) -> dict[str, int]:
    """Load the label vocabulary file used by the repository-level scripts."""
    label_file = Path(dataset_dir).resolve() / "labels.jsonl"
    with label_file.open("r", encoding="utf8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Label schema file is empty: {label_file}")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError(f"Label schema must be a JSON object: {label_file}")
    return {str(key): int(value) for key, value in payload.items()}

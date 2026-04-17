"""Readers for CoNLL BIO files and JSONL NER files with hard split isolation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import NERExample
from .split_guard import (
    ensure_dev_access,
    ensure_final_eval_access,
    ensure_test_text_only_access,
    ensure_train_access,
    normalize_split,
    placeholder_ner_tags,
)


JSON_VALUE_DECODER = json.JSONDecoder()
JSON_KEY_PATTERN = r'"{key}"\s*:'


def _finalize_example(
    examples: list[NERExample],
    tokens: list[str],
    ner_tags: list[str],
    split: str,
    source_path: str,
    has_labels: bool = True,
) -> None:
    if not tokens:
        return
    sample_id = f"{split}-{len(examples)}"
    examples.append(
        NERExample(
            sample_id=sample_id,
            tokens=list(tokens),
            ner_tags=list(ner_tags),
            split=split,
            source_path=source_path,
            has_labels=has_labels,
        )
    )


def _detect_input_format(path: str | Path, input_format: str) -> str:
    file_path = Path(path)
    detected_format = input_format
    if detected_format == "auto":
        detected_format = "jsonl" if file_path.suffix.lower() == ".jsonl" else "conll"
    if detected_format not in {"conll", "jsonl"}:
        raise ValueError(f"Unsupported input format: {detected_format}")
    return detected_format


def _extract_json_field(raw_line: str, key: str) -> object | None:
    match = re.search(JSON_KEY_PATTERN.format(key=re.escape(key)), raw_line)
    if match is None:
        return None
    value_start = match.end()
    while value_start < len(raw_line) and raw_line[value_start].isspace():
        value_start += 1
    value, _ = JSON_VALUE_DECODER.raw_decode(raw_line, idx=value_start)
    return value


def _coerce_text_to_tokens(value: object, file_path: Path, line_idx: int) -> list[str]:
    if isinstance(value, list):
        if not all(isinstance(token, str) for token in value):
            raise ValueError(f"'tokens' must be a list of strings at {file_path}:{line_idx}.")
        return [token for token in value if token]
    if isinstance(value, str):
        return [token for token in value.strip().split() if token]
    raise ValueError(
        f"Text-only JSONL records at {file_path}:{line_idx} must expose 'tokens', 'text', or 'sentence'."
    )


def read_conll_bio_file(path: str | Path, split: str) -> list[NERExample]:
    """Read a CoNLL-style BIO file into labeled NERExample objects."""
    file_path = Path(path)
    examples: list[NERExample] = []
    tokens: list[str] = []
    ner_tags: list[str] = []

    with file_path.open("r", encoding="utf8") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                _finalize_example(examples, tokens, ner_tags, split, str(file_path))
                tokens = []
                ner_tags = []
                continue
            if stripped.startswith("-DOCSTART-"):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed CoNLL line at {file_path}:{line_idx}: '{stripped}'")

            tokens.append(parts[0])
            ner_tags.append(parts[-1])

    _finalize_example(examples, tokens, ner_tags, split, str(file_path))
    return examples


def read_jsonl_ner_file(path: str | Path, split: str) -> list[NERExample]:
    """Read a JSONL NER file with `tokens` and `ner_tags` fields."""
    file_path = Path(path)
    examples: list[NERExample] = []

    with file_path.open("r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "tokens" not in record or "ner_tags" not in record:
                raise ValueError(
                    f"JSONL record at {file_path}:{idx + 1} must contain 'tokens' and 'ner_tags'."
                )
            if not isinstance(record["tokens"], list) or not all(isinstance(token, str) for token in record["tokens"]):
                raise ValueError(f"'tokens' must be a list of strings at {file_path}:{idx + 1}.")
            if not isinstance(record["ner_tags"], list) or not all(isinstance(tag, str) for tag in record["ner_tags"]):
                raise ValueError(f"'ner_tags' must be a list of strings at {file_path}:{idx + 1}.")
            examples.append(
                NERExample(
                    sample_id=str(record.get("sample_id", f"{split}-{idx}")),
                    tokens=list(record["tokens"]),
                    ner_tags=list(record["ner_tags"]),
                    split=str(record.get("split", split)),
                    source_path=str(file_path.resolve()),
                    has_labels=True,
                )
            )
    return examples


def read_jsonl_text_only_file(path: str | Path, split: str) -> list[NERExample]:
    """Read only test text from JSONL without materializing label fields."""
    file_path = Path(path)
    examples: list[NERExample] = []

    with file_path.open("r", encoding="utf8") as f:
        for idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            tokens_source = _extract_json_field(stripped, "tokens")
            if tokens_source is None:
                tokens_source = _extract_json_field(stripped, "text")
            if tokens_source is None:
                tokens_source = _extract_json_field(stripped, "sentence")
            if tokens_source is None:
                raise ValueError(
                    f"Text-only JSONL record at {file_path}:{idx} must contain 'tokens', 'text', or 'sentence'."
                )

            tokens = _coerce_text_to_tokens(tokens_source, file_path, idx)
            if not tokens:
                continue

            sample_id = _extract_json_field(stripped, "sample_id")
            record_split = _extract_json_field(stripped, "split")
            examples.append(
                NERExample(
                    sample_id=str(sample_id) if sample_id is not None else f"{split}-{len(examples)}",
                    tokens=tokens,
                    ner_tags=placeholder_ner_tags(len(tokens)),
                    split=str(record_split) if isinstance(record_split, str) else split,
                    source_path=str(file_path.resolve()),
                    has_labels=False,
                )
            )
    return examples


def _load_labeled_ner_examples(
    path: str | Path,
    input_format: str,
    split: str,
    max_samples: int | None = None,
) -> list[NERExample]:
    detected_format = _detect_input_format(path, input_format)
    if detected_format == "conll":
        examples = read_conll_bio_file(path, split)
    else:
        examples = read_jsonl_ner_file(path, split)

    if max_samples is not None:
        return examples[: max(0, int(max_samples))]
    return examples


def load_train_data(
    path: str | Path,
    input_format: str,
    max_samples: int | None = None,
    stage: str = "train",
) -> list[NERExample]:
    """Load labeled train data. Only valid during stage='train'."""
    ensure_train_access(stage, operation="load_train_data()", source_path=str(Path(path)))
    return _load_labeled_ner_examples(path, input_format=input_format, split="train", max_samples=max_samples)


def load_dev_data(
    path: str | Path,
    input_format: str,
    max_samples: int | None = None,
    stage: str = "dev",
    split_name: str = "validation",
) -> list[NERExample]:
    """Load labeled dev data. Only valid during stage='dev'."""
    ensure_dev_access(stage, operation="load_dev_data()", source_path=str(Path(path)))
    normalized_split = normalize_split(split_name)
    if normalized_split != "dev":
        raise ValueError("load_dev_data() may only be used for the dev/validation split.")
    return _load_labeled_ner_examples(path, input_format=input_format, split=normalized_split, max_samples=max_samples)


def load_test_text_only(
    path: str | Path,
    input_format: str,
    max_samples: int | None = None,
    stage: str = "test_infer",
    split_name: str = "test",
) -> list[NERExample]:
    """Load test text without labels. Only valid during stage='test_infer'."""
    ensure_test_text_only_access(stage, operation="load_test_text_only()", source_path=str(Path(path)))
    normalized_split = normalize_split(split_name)
    if normalized_split != "test":
        raise ValueError("load_test_text_only() may only be used for the test split.")

    detected_format = _detect_input_format(path, input_format)
    if detected_format != "jsonl":
        raise ValueError(
            "test_infer requires a JSONL source with accessible text-only fields. "
            "Labeled CoNLL test files are forbidden because they expose gold tags."
        )

    examples = read_jsonl_text_only_file(path, split=normalized_split)
    if max_samples is not None:
        return examples[: max(0, int(max_samples))]
    return examples


def load_test_with_labels_for_final_eval(
    path: str | Path,
    input_format: str,
    max_samples: int | None = None,
    stage: str = "final_eval",
    split_name: str = "test",
) -> list[NERExample]:
    """Load labeled test data. Only valid during stage='final_eval'."""
    ensure_final_eval_access(
        stage,
        operation="load_test_with_labels_for_final_eval()",
        source_path=str(Path(path)),
    )
    normalized_split = normalize_split(split_name)
    if normalized_split != "test":
        raise ValueError("load_test_with_labels_for_final_eval() may only be used for the test split.")
    return _load_labeled_ner_examples(path, input_format=input_format, split=normalized_split, max_samples=max_samples)


def load_ner_examples(
    path: str | Path,
    input_format: str,
    split: str,
    max_samples: int | None = None,
) -> list[NERExample]:
    """Disabled generic loader. Use explicit split-specific loaders instead."""
    raise RuntimeError(
        "load_ner_examples() is disabled. "
        "Use load_train_data(), load_dev_data(), load_test_text_only(), "
        "or load_test_with_labels_for_final_eval()."
    )

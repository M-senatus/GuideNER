"""Readers for CoNLL BIO files and JSONL NER files."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import NERExample


def _finalize_example(
    examples: list[NERExample],
    tokens: list[str],
    ner_tags: list[str],
    split: str,
    source_path: str,
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
        )
    )


def read_conll_bio_file(path: str | Path, split: str) -> list[NERExample]:
    """Read a CoNLL-style BIO file into NERExample objects."""
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
                    source_path=str(file_path),
                )
            )
    return examples


def load_ner_examples(
    path: str | Path,
    input_format: str,
    split: str,
    max_samples: int | None = None,
) -> list[NERExample]:
    """Load NER examples from disk and optionally cap the sample count."""
    file_path = Path(path)
    detected_format = input_format
    if detected_format == "auto":
        detected_format = "jsonl" if file_path.suffix.lower() == ".jsonl" else "conll"

    if detected_format == "conll":
        examples = read_conll_bio_file(file_path, split)
    elif detected_format == "jsonl":
        examples = read_jsonl_ner_file(file_path, split)
    else:
        raise ValueError(f"Unsupported input format: {detected_format}")

    if max_samples is not None:
        return examples[: max(0, int(max_samples))]
    return examples

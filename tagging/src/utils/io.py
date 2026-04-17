"""Small I/O helpers used across the tagging pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_json(path: str | Path) -> dict:
    """Read a JSON file into a dictionary."""
    with Path(path).open("r", encoding="utf8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping, indent: int = 2) -> Path:
    """Write a dictionary to JSON with UTF-8 encoding."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(path: str | Path, records: Sequence[Mapping] | Iterable[Mapping]) -> Path:
    """Write JSONL records to disk."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf8") as f:
        for record in records:
            f.write(json.dumps(dict(record), ensure_ascii=False))
            f.write("\n")
    return output_path

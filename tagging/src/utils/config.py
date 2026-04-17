"""Configuration loading and validation utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import read_json


REQUIRED_TOP_LEVEL_KEYS = {"experiment", "data", "model", "training", "inference", "export"}
REQUIRED_DATA_KEYS = {"train_path", "validation_path", "test_path", "format", "label_scheme"}
REQUIRED_MODEL_KEYS = {"name_or_path", "max_length"}
REQUIRED_TRAINING_KEYS = {
    "output_dir",
    "num_train_epochs",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "learning_rate",
}
REQUIRED_EXPORT_KEYS = {"output_dir", "batch_size", "hidden_state_layer", "vector_type"}
PATH_FIELDS = {
    ("data", "train_path"),
    ("data", "validation_path"),
    ("data", "test_path"),
    ("training", "output_dir"),
    ("export", "output_dir"),
}


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Apply dotted-key overrides to a nested configuration dictionary."""
    updated = deepcopy(config)
    if overrides is None:
        return updated

    for key, value in overrides.items():
        if value is None:
            continue
        _set_nested(updated, key, value)
    return updated


def _resolve_known_paths(config: dict[str, Any], tagging_root: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    for section, key in PATH_FIELDS:
        raw_value = resolved[section][key]
        if raw_value is None:
            continue
        path_value = Path(raw_value)
        if not path_value.is_absolute():
            path_value = (tagging_root / path_value).resolve()
        resolved[section][key] = str(path_value)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """Validate required sections and a few critical fields."""
    missing_top_level = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    if missing_top_level:
        raise ValueError(f"Missing top-level config keys: {sorted(missing_top_level)}")

    for key_set, section_name in [
        (REQUIRED_DATA_KEYS, "data"),
        (REQUIRED_MODEL_KEYS, "model"),
        (REQUIRED_TRAINING_KEYS, "training"),
        (REQUIRED_EXPORT_KEYS, "export"),
    ]:
        missing = key_set - set(config[section_name].keys())
        if missing:
            raise ValueError(f"Missing keys in '{section_name}': {sorted(missing)}")

    if config["data"]["format"] not in {"conll", "jsonl", "auto"}:
        raise ValueError("data.format must be one of: conll, jsonl, auto")
    if config["data"]["label_scheme"].upper() != "BIO":
        raise ValueError("This implementation currently supports BIO labels only.")
    if int(config["model"]["max_length"]) <= 0:
        raise ValueError("model.max_length must be positive.")
    if int(config["training"]["per_device_train_batch_size"]) <= 0:
        raise ValueError("training.per_device_train_batch_size must be positive.")
    if int(config["training"]["per_device_eval_batch_size"]) <= 0:
        raise ValueError("training.per_device_eval_batch_size must be positive.")
    if float(config["training"]["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive.")


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load, resolve, override, and validate a tagging configuration."""
    config_file = Path(config_path).resolve()
    tagging_root = config_file.parents[1]
    config = read_json(config_file)
    config = apply_overrides(config, overrides)
    config = _resolve_known_paths(config, tagging_root)
    validate_config(config)
    config["_meta"] = {
        "config_path": str(config_file),
        "tagging_root": str(tagging_root),
    }
    return config

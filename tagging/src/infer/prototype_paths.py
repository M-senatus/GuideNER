"""Helpers for resolving shared guideline prototype storage paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def default_prototype_dir_from_config(config: Mapping[str, Any]) -> Path:
    """Return the repository-level prototype directory for the current model and dataset."""
    tagging_root = Path(str(config["_meta"]["tagging_root"])).resolve()
    project_root = tagging_root.parent
    model_name = Path(str(config["model"]["name_or_path"])).name
    dataset_name = Path(str(config["data"]["train_path"])).resolve().parent.name
    return project_root / "prototypes" / f"{model_name}-{dataset_name}-prototypes"

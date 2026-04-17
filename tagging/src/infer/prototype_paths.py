"""Helpers for resolving shared guideline prototype storage paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _project_root_from_config(config: Mapping[str, Any]) -> Path:
    """Return the repository root inferred from the tagging config metadata."""
    tagging_root = Path(str(config["_meta"]["tagging_root"])).resolve()
    return tagging_root.parent


def dataset_name_from_config(config: Mapping[str, Any]) -> str:
    """Return the dataset directory name used by the tagging config."""
    return Path(str(config["data"]["train_path"])).resolve().parent.name


def prototype_dir_from_model_and_dataset(
    config: Mapping[str, Any],
    model_name: str,
    dataset_name: str | None = None,
) -> Path:
    """Return GuideNER/prototypes/{llm-model}-{dataset}-prototypes."""
    project_root = _project_root_from_config(config)
    resolved_dataset_name = dataset_name or dataset_name_from_config(config)
    if not str(model_name).strip():
        raise ValueError("Prototype model name must be a non-empty string.")
    return project_root / "prototypes" / f"{model_name}-{resolved_dataset_name}-prototypes"


def infer_model_name_from_guideline_path(guideline_path: str | Path) -> str:
    """Infer the LLM model name from a guideline file like {model_name}_summaryrules.json."""
    guideline_name = Path(guideline_path).stem
    summary_suffix = "_summaryrules"
    if guideline_name.endswith(summary_suffix):
        inferred_name = guideline_name[: -len(summary_suffix)]
        if inferred_name:
            return inferred_name
    raise ValueError(
        "Could not infer prototype model name from guideline path. "
        "Expected a filename like 'Llama-3.1-8B-Instruct_summaryrules.json'."
    )


def default_prototype_dir_from_guideline_path(config: Mapping[str, Any], guideline_path: str | Path) -> Path:
    """Return the default prototype directory using the guideline-producing LLM name."""
    model_name = infer_model_name_from_guideline_path(guideline_path)
    return prototype_dir_from_model_and_dataset(config, model_name=model_name)

"""Metrics for token classification and span-level NER evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
except ImportError as exc:  # pragma: no cover - dependency checked in runtime environments
    _SEQEVAL_IMPORT_ERROR = exc
    accuracy_score = None
    f1_score = None
    precision_score = None
    recall_score = None
else:
    _SEQEVAL_IMPORT_ERROR = None


def ensure_seqeval_available(task_name: str = "NER evaluation") -> None:
    """Raise a clear error when the optional seqeval dependency is missing."""
    if _SEQEVAL_IMPORT_ERROR is not None:
        raise ImportError(
            f"seqeval is required for {task_name}. "
            "Install it with `pip install seqeval` or `pip install -r tagging/requirements.txt`."
        ) from _SEQEVAL_IMPORT_ERROR


def logits_to_label_sequences(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert logits and label ids into seqeval-compatible string sequences."""
    predicted_ids = predictions.argmax(axis=-1)
    true_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []

    for pred_row, label_row in zip(predicted_ids, labels):
        true_tags: list[str] = []
        pred_tags: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if int(label_id) == -100:
                continue
            true_tags.append(id2label[int(label_id)])
            pred_tags.append(id2label[int(pred_id)])
        true_sequences.append(true_tags)
        pred_sequences.append(pred_tags)

    return true_sequences, pred_sequences


def compute_seqeval_metrics_from_sequences(
    true_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, float]:
    """Compute standard span-level NER metrics from tag sequences."""
    ensure_seqeval_available()
    return {
        "precision": float(precision_score(true_sequences, pred_sequences)),
        "recall": float(recall_score(true_sequences, pred_sequences)),
        "f1": float(f1_score(true_sequences, pred_sequences)),
        "accuracy": float(accuracy_score(true_sequences, pred_sequences)),
    }


def build_compute_metrics(id2label: dict[int, str]):
    """Build a Trainer-compatible metric callback."""
    # Fail before training starts instead of surfacing a missing metric dependency
    # only when the first evaluation step runs.
    ensure_seqeval_available(task_name="training-time NER evaluation")

    def _compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions, labels = eval_prediction
        true_sequences, pred_sequences = logits_to_label_sequences(predictions, labels, id2label)
        return compute_seqeval_metrics_from_sequences(true_sequences, pred_sequences)

    return _compute_metrics

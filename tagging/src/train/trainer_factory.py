"""Factory helpers for Hugging Face Trainer objects."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from ..utils.io import ensure_dir


def create_training_arguments(config: dict[str, Any]) -> TrainingArguments:
    """Create TrainingArguments with reproducible defaults for NER research."""
    training_cfg = config["training"]
    experiment_cfg = config["experiment"]
    output_root = Path(training_cfg["output_dir"])
    checkpoint_dir = ensure_dir(output_root / "checkpoints")
    ensure_dir(output_root / "logs")

    signature = inspect.signature(TrainingArguments.__init__)
    kwargs: dict[str, Any] = dict(
        output_dir=str(checkpoint_dir),
        logging_dir=str(output_root / "logs"),
        overwrite_output_dir=bool(training_cfg.get("overwrite_output_dir", False)),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
        logging_steps=int(training_cfg.get("logging_steps", 50)),
        save_strategy=str(training_cfg.get("save_strategy", "epoch")),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(training_cfg.get("metric_for_best_model", "f1")),
        greater_is_better=bool(training_cfg.get("greater_is_better", True)),
        report_to=list(training_cfg.get("report_to", [])),
        fp16=bool(training_cfg.get("fp16", False)),
        seed=int(experiment_cfg["seed"]),
        data_seed=int(experiment_cfg["seed"]),
    )

    eval_strategy_value = str(training_cfg.get("evaluation_strategy", "epoch"))
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = eval_strategy_value
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy_value
    else:
        raise TypeError(
            "This transformers version does not expose either 'evaluation_strategy' or 'eval_strategy' "
            "in TrainingArguments."
        )

    return TrainingArguments(**kwargs)


def create_trainer(
    model: Any,
    args: TrainingArguments,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    data_collator: Any,
    compute_metrics: Any,
    config: dict[str, Any],
) -> Trainer:
    """Assemble a Trainer with optional early stopping."""
    callbacks: list[Any] = []
    patience = config["training"].get("early_stopping_patience")
    if patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

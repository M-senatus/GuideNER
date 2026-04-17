"""Training entrypoint for DeBERTa-based NER fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..data.collators import get_token_classification_collator
from ..data.labeling import build_label_mappings, validate_example_labels
from ..data.readers import load_ner_examples
from ..data.tokenization import examples_to_dataset, tokenize_and_align_labels
from ..models.deberta_token_classifier import load_token_classification_model, load_tokenizer
from ..train.metrics import build_compute_metrics
from ..train.trainer_factory import create_trainer, create_training_arguments
from ..utils.config import load_config
from ..utils.io import ensure_dir, write_json
from ..utils.seed import set_global_seed


def _resolve_output_paths(output_root: Path) -> dict[str, Path]:
    """Create the standard output tree for an experiment."""
    return {
        "root": ensure_dir(output_root),
        "artifacts": ensure_dir(output_root / "artifacts"),
        "eval": ensure_dir(output_root / "eval"),
        "predictions": ensure_dir(output_root / "predictions"),
        "checkpoint_best": ensure_dir(output_root / "checkpoint-best"),
    }


def _load_splits(config: dict[str, Any]) -> tuple[list, list, list]:
    """Load train, validation, and test examples from disk."""
    data_cfg = config["data"]
    train_examples = load_ner_examples(
        data_cfg["train_path"],
        input_format=data_cfg["format"],
        split="train",
        max_samples=data_cfg.get("max_train_samples"),
    )
    validation_examples = load_ner_examples(
        data_cfg["validation_path"],
        input_format=data_cfg["format"],
        split="validation",
        max_samples=data_cfg.get("max_validation_samples"),
    )
    test_examples = load_ner_examples(
        data_cfg["test_path"],
        input_format=data_cfg["format"],
        split="test",
        max_samples=data_cfg.get("max_test_samples"),
    )
    return train_examples, validation_examples, test_examples


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train a DeBERTa token-classification NER model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir.")
    parser.add_argument("--model-name-or-path", default=None, help="Override model.name_or_path.")
    parser.add_argument("--max-length", type=int, default=None, help="Override model.max_length.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override training.learning_rate.")
    parser.add_argument("--num-train-epochs", type=float, default=None, help="Override training.num_train_epochs.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end NER fine-tuning pipeline."""
    args = parse_args()
    overrides = {
        "training.output_dir": args.output_dir,
        "model.name_or_path": args.model_name_or_path,
        "model.max_length": args.max_length,
        "training.learning_rate": args.learning_rate,
        "training.num_train_epochs": args.num_train_epochs,
        "training.per_device_train_batch_size": args.per_device_train_batch_size,
        "training.per_device_eval_batch_size": args.per_device_eval_batch_size,
        "data.max_train_samples": args.max_train_samples,
        "data.max_validation_samples": args.max_validation_samples,
        "data.max_test_samples": args.max_test_samples,
    }
    config = load_config(args.config, overrides=overrides)
    set_global_seed(int(config["experiment"]["seed"]))

    output_paths = _resolve_output_paths(Path(config["training"]["output_dir"]))
    write_json(output_paths["artifacts"] / "resolved_config.json", config)

    train_examples, validation_examples, test_examples = _load_splits(config)
    labels, label2id, id2label = build_label_mappings(train_examples)
    validate_example_labels(validation_examples, set(labels))
    validate_example_labels(test_examples, set(labels))

    write_json(output_paths["artifacts"] / "labels.json", {"labels": labels})
    write_json(output_paths["artifacts"] / "label2id.json", label2id)
    write_json(output_paths["artifacts"] / "id2label.json", {str(k): v for k, v in id2label.items()})

    tokenizer = load_tokenizer(config["model"]["name_or_path"])
    model = load_token_classification_model(
        config["model"]["name_or_path"],
        label2id=label2id,
        id2label=id2label,
        output_hidden_states=False,
    )

    max_length = int(config["model"]["max_length"])
    train_dataset = tokenize_and_align_labels(examples_to_dataset(train_examples), tokenizer, label2id, max_length)
    validation_dataset = tokenize_and_align_labels(
        examples_to_dataset(validation_examples),
        tokenizer,
        label2id,
        max_length,
    )
    test_dataset = tokenize_and_align_labels(examples_to_dataset(test_examples), tokenizer, label2id, max_length)

    training_args = create_training_arguments(config)
    trainer = create_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=get_token_classification_collator(tokenizer),
        compute_metrics=build_compute_metrics(id2label),
        config=config,
    )

    train_result = trainer.train()
    trainer.save_state()
    write_json(output_paths["artifacts"] / "training_args.json", training_args.to_dict())
    write_json(output_paths["artifacts"] / "train_metrics.json", train_result.metrics)
    trainer.state.save_to_json(str(output_paths["artifacts"] / "trainer_state.json"))

    eval_metrics = trainer.evaluate(eval_dataset=validation_dataset, metric_key_prefix="eval")
    write_json(output_paths["eval"] / "eval_metrics.json", eval_metrics)

    test_output = trainer.predict(test_dataset, metric_key_prefix="test")
    write_json(output_paths["eval"] / "test_metrics.json", test_output.metrics)

    trainer.save_model(str(output_paths["checkpoint_best"]))
    tokenizer.save_pretrained(str(output_paths["checkpoint_best"]))
    write_json(
        output_paths["artifacts"] / "run_summary.json",
        {
            "best_checkpoint_dir": str(output_paths["checkpoint_best"]),
            "trainer_best_model_checkpoint": trainer.state.best_model_checkpoint,
            "num_train_examples": len(train_examples),
            "num_validation_examples": len(validation_examples),
            "num_test_examples": len(test_examples),
        },
    )


if __name__ == "__main__":
    main()

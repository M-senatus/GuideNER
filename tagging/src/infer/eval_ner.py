"""Evaluation entrypoint built on top of the prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..data.readers import load_ner_examples
from ..infer.predict import compute_metrics_from_prediction_records, predict_examples
from ..models.deberta_token_classifier import load_checkpoint_model, load_tokenizer
from ..utils.config import load_config
from ..utils.io import ensure_dir, write_json, write_jsonl


def _resolve_data_path(config: dict[str, Any], split: str | None, input_path: str | None) -> tuple[str, str]:
    """Resolve the evaluation data source."""
    if input_path is not None:
        return str(Path(input_path).resolve()), "auto"
    if split is None:
        raise ValueError("Either --split or --input-path must be provided.")

    split_map = {
        "train": "train_path",
        "validation": "validation_path",
        "dev": "validation_path",
        "test": "test_path",
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split '{split}'.")
    return config["data"][split_map[split]], config["data"]["format"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned NER checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint-path", required=True, help="Fine-tuned checkpoint directory.")
    parser.add_argument("--split", default="test", help="Named split to evaluate: train/validation/test.")
    parser.add_argument("--input-path", default=None, help="Optional explicit input file path.")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation outputs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference.batch_size.")
    return parser.parse_args()


def main() -> None:
    """Run evaluation and save both predictions and metrics."""
    args = parse_args()
    config = load_config(args.config, overrides={"inference.batch_size": args.batch_size})
    data_path, input_format = _resolve_data_path(config, split=args.split, input_path=args.input_path)
    examples = load_ner_examples(data_path, input_format=input_format, split=args.split)

    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    tokenizer = load_tokenizer(checkpoint_path)
    model = load_checkpoint_model(checkpoint_path, output_hidden_states=False)

    prediction_records = predict_examples(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=int(config["inference"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
    )
    metrics = compute_metrics_from_prediction_records(prediction_records)

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["training"]["output_dir"]) / "eval" / args.split
    ensure_dir(output_dir)
    write_jsonl(output_dir / "predictions.jsonl", prediction_records)
    write_json(output_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()

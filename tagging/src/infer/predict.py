"""Prediction helpers for NER inference and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency checked in runtime environments
    raise ImportError("PyTorch is required for prediction.") from exc

from ..data.collators import ExportCollator
from ..data.readers import load_ner_examples
from ..data.tokenization import build_inference_features
from ..models.deberta_token_classifier import load_checkpoint_model, load_tokenizer
from ..train.metrics import compute_seqeval_metrics_from_sequences, ensure_seqeval_available
from ..utils.config import load_config
from ..utils.io import ensure_dir, write_json, write_jsonl


def _resolve_data_path(config: dict[str, Any], split: str | None, input_path: str | None) -> tuple[str, str]:
    """Resolve the input data source for prediction."""
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


def _get_device() -> torch.device:
    """Select a CUDA device when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move only model tensor inputs onto the target device."""
    model_inputs: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in batch:
            model_inputs[key] = batch[key].to(device)
    return model_inputs


def predict_examples(
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    batch_size: int,
    max_length: int,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Run word-level prediction while preserving subword-level debug metadata."""
    if device is None:
        device = _get_device()

    model.to(device)
    model.eval()

    features = build_inference_features(examples, tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(features, batch_size=batch_size, shuffle=False, collate_fn=ExportCollator(tokenizer))
    id2label = {int(key): value for key, value in model.config.id2label.items()}

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting NER", leave=False):
            outputs = model(**_build_model_inputs(batch, device))
            logits = outputs.logits.detach().cpu()
            input_ids = batch["input_ids"].detach().cpu()
            attention_mask = batch["attention_mask"].detach().cpu()

            for idx, sample_id in enumerate(batch["sample_id"]):
                sequence_length = int(attention_mask[idx].sum().item())
                sample_logits = logits[idx, :sequence_length].numpy()
                sample_input_ids = input_ids[idx, :sequence_length].tolist()
                sample_word_ids = batch["word_ids"][idx][:sequence_length]
                sample_tokens = batch["tokens"][idx]
                sample_gold_tags = batch["ner_tags"][idx]

                subword_predictions: list[dict[str, Any]] = []
                word_predictions: list[str] = []
                seen_word_ids: set[int] = set()

                for subword_index, (token_id, word_id, pred_id) in enumerate(
                    zip(sample_input_ids, sample_word_ids, sample_logits.argmax(axis=-1).tolist())
                ):
                    if word_id < 0:
                        continue
                    pred_label = id2label[int(pred_id)]
                    subword_predictions.append(
                        {
                            "subword_index": subword_index,
                            "word_index": word_id,
                            "subword_text": tokenizer.convert_ids_to_tokens(int(token_id)),
                            "pred_label": pred_label,
                        }
                    )
                    if word_id not in seen_word_ids:
                        word_predictions.append(pred_label)
                        seen_word_ids.add(word_id)

                visible_length = len(word_predictions)
                records.append(
                    {
                        "sample_id": sample_id,
                        "split": batch["split"][idx],
                        "source_path": batch["source_path"][idx],
                        "tokens": list(sample_tokens),
                        "gold_tags": list(sample_gold_tags),
                        "visible_tokens": list(sample_tokens[:visible_length]),
                        "visible_gold_tags": list(sample_gold_tags[:visible_length]),
                        "pred_tags": word_predictions,
                        "truncated": visible_length < len(sample_tokens),
                        "subword_predictions": subword_predictions,
                    }
                )
    return records


def compute_metrics_from_prediction_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute seqeval metrics from prediction records with gold labels."""
    true_sequences = [record["visible_gold_tags"] for record in records]
    pred_sequences = [record["pred_tags"] for record in records]
    return compute_seqeval_metrics_from_sequences(true_sequences, pred_sequences)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone prediction."""
    parser = argparse.ArgumentParser(description="Run NER prediction from a fine-tuned checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint-path", required=True, help="Fine-tuned checkpoint directory.")
    parser.add_argument("--split", default="test", help="Named split to predict: train/validation/test.")
    parser.add_argument("--input-path", default=None, help="Optional explicit input file path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference.batch_size.")
    parser.add_argument("--output-dir", default=None, help="Directory for prediction outputs.")
    parser.add_argument("--compute-metrics", action="store_true", help="Compute seqeval metrics when gold tags exist.")
    return parser.parse_args()


def main() -> None:
    """Run standalone prediction and optionally compute metrics."""
    args = parse_args()
    if args.compute_metrics:
        ensure_seqeval_available(task_name="prediction-time NER evaluation")
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

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["training"]["output_dir"]) / "predictions"
    ensure_dir(output_dir)
    write_jsonl(output_dir / "predictions.jsonl", prediction_records)
    if args.compute_metrics:
        write_json(output_dir / "metrics.json", compute_metrics_from_prediction_records(prediction_records))


if __name__ == "__main__":
    main()

"""Export token-, word-, and span-level vectors from a fine-tuned NER checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency checked in runtime environments
    raise ImportError("PyTorch is required for hidden-state export.") from exc

from ..data.collators import ExportCollator
from ..data.labeling import extract_spans_from_bio
from ..data.readers import load_ner_examples
from ..data.schemas import SpanAnnotation
from ..data.tokenization import build_inference_features
from ..models.deberta_token_classifier import load_checkpoint_model, load_tokenizer
from ..utils.config import load_config
from ..utils.io import ensure_dir, write_json, write_jsonl


def _resolve_data_path(config: dict[str, Any], split: str | None, input_path: str | None) -> tuple[str, str]:
    """Resolve the input data source for vector export."""
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


def _load_span_file(span_file: str | None) -> dict[str, list[SpanAnnotation]]:
    """Load optional explicit spans from a JSONL file."""
    if span_file is None:
        return {}

    span_map: dict[str, list[SpanAnnotation]] = {}
    with Path(span_file).open("r", encoding="utf8") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            for required_key in ("sample_id", "start", "end"):
                if required_key not in record:
                    raise ValueError(f"Span file line {line_idx} is missing '{required_key}'.")
            span = SpanAnnotation(
                sample_id=str(record["sample_id"]),
                start=int(record["start"]),
                end=int(record["end"]),
                entity_type=record.get("entity_type"),
                metadata={
                    key: value
                    for key, value in record.items()
                    if key not in {"sample_id", "start", "end", "entity_type"}
                },
            )
            span_map.setdefault(span.sample_id, []).append(span)
    return span_map


def _get_device() -> torch.device:
    """Select a CUDA device when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move model input tensors onto the selected device."""
    model_inputs: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in batch:
            model_inputs[key] = batch[key].to(device)
    return model_inputs


def _pool_word_vectors(sample_hidden: np.ndarray, word_ids: list[int], valid_length: int) -> dict[int, np.ndarray]:
    """Average subword vectors into word-level vectors."""
    grouped: dict[int, list[np.ndarray]] = {}
    for token_index in range(valid_length):
        word_id = word_ids[token_index]
        if word_id < 0:
            continue
        grouped.setdefault(word_id, []).append(sample_hidden[token_index])
    return {
        word_id: np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        for word_id, vectors in grouped.items()
    }


def export_vectors(
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    batch_size: int,
    max_length: int,
    output_dir: str | Path,
    checkpoint_path: str,
    vector_type: str,
    hidden_state_layer: int,
    span_file: str | None = None,
) -> dict[str, Any]:
    """Export token, word, or span vectors plus retrieval-ready metadata."""
    device = _get_device()
    model.to(device)
    model.eval()

    example_map = {example.sample_id: example for example in examples}
    explicit_spans = _load_span_file(span_file)
    features = build_inference_features(examples, tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(features, batch_size=batch_size, shuffle=False, collate_fn=ExportCollator(tokenizer))

    output_root = ensure_dir(output_dir)
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    skipped_spans = 0
    hidden_size = int(model.config.hidden_size)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Exporting {vector_type} vectors", leave=False):
            outputs = model(**_build_model_inputs(batch, device), output_hidden_states=True)
            hidden_states = outputs.hidden_states[hidden_state_layer].detach().cpu().numpy()
            input_ids = batch["input_ids"].detach().cpu()
            attention_mask = batch["attention_mask"].detach().cpu()

            for batch_index, sample_id in enumerate(batch["sample_id"]):
                sequence_length = int(attention_mask[batch_index].sum().item())
                sample_hidden = hidden_states[batch_index, :sequence_length]
                sample_input_ids = input_ids[batch_index, :sequence_length].tolist()
                sample_word_ids = batch["word_ids"][batch_index][:sequence_length]
                sample_tokens = batch["tokens"][batch_index]
                sample_gold_tags = batch["ner_tags"][batch_index]
                word_vectors = _pool_word_vectors(sample_hidden, sample_word_ids, sequence_length)

                if vector_type == "token":
                    for token_index, (token_id, word_id) in enumerate(zip(sample_input_ids, sample_word_ids)):
                        if word_id < 0:
                            continue
                        vectors.append(sample_hidden[token_index].astype(np.float32))
                        metadata.append(
                            {
                                "sample_id": sample_id,
                                "split": batch["split"][batch_index],
                                "source_path": batch["source_path"][batch_index],
                                "checkpoint_path": checkpoint_path,
                                "vector_type": "token",
                                "token_index": token_index,
                                "word_index": word_id,
                                "token_text": tokenizer.convert_ids_to_tokens(int(token_id)),
                                "word_text": sample_tokens[word_id],
                                "tokens": sample_tokens,
                                "words": sample_tokens,
                                "label_sequence": sample_gold_tags,
                                "entity_type": sample_gold_tags[word_id],
                            }
                        )
                elif vector_type == "word":
                    for word_index in sorted(word_vectors.keys()):
                        vectors.append(word_vectors[word_index].astype(np.float32))
                        metadata.append(
                            {
                                "sample_id": sample_id,
                                "split": batch["split"][batch_index],
                                "source_path": batch["source_path"][batch_index],
                                "checkpoint_path": checkpoint_path,
                                "vector_type": "word",
                                "word_index": word_index,
                                "word_text": sample_tokens[word_index],
                                "tokens": sample_tokens,
                                "words": sample_tokens,
                                "label_sequence": sample_gold_tags,
                                "entity_type": sample_gold_tags[word_index],
                            }
                        )
                elif vector_type == "span":
                    example = example_map[sample_id]
                    spans = explicit_spans.get(sample_id, extract_spans_from_bio(example))
                    for span in spans:
                        if any(word_index not in word_vectors for word_index in range(span.start, span.end)):
                            skipped_spans += 1
                            continue
                        span_vector = np.mean(
                            np.stack([word_vectors[word_index] for word_index in range(span.start, span.end)], axis=0),
                            axis=0,
                        ).astype(np.float32)
                        vectors.append(span_vector)
                        metadata.append(
                            {
                                "sample_id": sample_id,
                                "split": batch["split"][batch_index],
                                "source_path": batch["source_path"][batch_index],
                                "checkpoint_path": checkpoint_path,
                                "vector_type": "span",
                                "span_start": span.start,
                                "span_end": span.end,
                                "span_text": span.metadata.get("span_text", " ".join(sample_tokens[span.start:span.end])),
                                "tokens": sample_tokens,
                                "words": sample_tokens,
                                "label_sequence": sample_gold_tags,
                                "entity_type": span.entity_type,
                            }
                        )
                else:
                    raise ValueError(f"Unsupported vector_type: {vector_type}")

    vector_array = (
        np.stack(vectors, axis=0).astype(np.float32)
        if vectors
        else np.zeros((0, hidden_size), dtype=np.float32)
    )
    np.save(output_root / "vectors.npy", vector_array)
    write_jsonl(output_root / "metadata.jsonl", metadata)
    summary = {
        "vector_type": vector_type,
        "vector_count": int(vector_array.shape[0]),
        "hidden_size": hidden_size,
        "checkpoint_path": checkpoint_path,
        "skipped_spans": skipped_spans,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for vector export."""
    parser = argparse.ArgumentParser(description="Export token, word, or span vectors from a NER checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint-path", required=True, help="Fine-tuned checkpoint directory.")
    parser.add_argument("--split", default="test", help="Named split to export: train/validation/test.")
    parser.add_argument("--input-path", default=None, help="Optional explicit input file path.")
    parser.add_argument("--span-file", default=None, help="Optional JSONL span file for span export.")
    parser.add_argument("--vector-type", choices=["token", "word", "span"], default=None)
    parser.add_argument("--output-dir", default=None, help="Override export.output_dir.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override export.batch_size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for export examples.")
    parser.add_argument("--hidden-state-layer", type=int, default=None, help="Layer index to export, default last layer.")
    return parser.parse_args()


def main() -> None:
    """Run hidden-state export from the CLI."""
    args = parse_args()
    overrides = {
        "export.output_dir": args.output_dir,
        "export.batch_size": args.batch_size,
        "export.max_samples": args.max_samples,
        "export.hidden_state_layer": args.hidden_state_layer,
        "export.vector_type": args.vector_type,
    }
    config = load_config(args.config, overrides=overrides)
    data_path, input_format = _resolve_data_path(config, split=args.split, input_path=args.input_path)
    examples = load_ner_examples(
        data_path,
        input_format=input_format,
        split=args.split,
        max_samples=config["export"].get("max_samples"),
    )

    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    tokenizer = load_tokenizer(checkpoint_path)
    model = load_checkpoint_model(checkpoint_path, output_hidden_states=True)

    vector_type = config["export"]["vector_type"]
    export_dir = Path(config["export"]["output_dir"]) / vector_type / (args.split or "custom")
    export_vectors(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=int(config["export"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        output_dir=export_dir,
        checkpoint_path=checkpoint_path,
        vector_type=vector_type,
        hidden_state_layer=int(config["export"]["hidden_state_layer"]),
        span_file=args.span_file,
    )


if __name__ == "__main__":
    main()

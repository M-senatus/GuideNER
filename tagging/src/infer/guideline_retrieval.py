"""Shared helpers for offline guideline prototype building and online retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency checked in runtime environments
    raise ImportError("PyTorch is required for guideline retrieval.") from exc

from ..data.collators import ExportCollator
from ..data.readers import load_dev_data, load_test_text_only, load_test_with_labels_for_final_eval, load_train_data
from ..data.schemas import NERExample
from ..data.split_guard import ensure_dev_access, ensure_train_access, normalize_split, normalize_stage
from ..data.tokenization import build_inference_features
from ..models.deberta_token_classifier import load_checkpoint_model, load_tokenizer
from ..utils.config import load_config
from ..utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


GUIDELINE_TYPE_KEYS = ("entity_type", "type", "label")
GUIDELINE_RULE_KEYS = ("rule", "pattern", "guideline", "description")
GUIDELINE_INSTANCE_KEYS = (
    "entity_instances",
    "instances",
    "support_examples",
    "examples",
    "entities",
)


def _resolve_data_path(config: dict[str, Any], split: str | None, input_path: str | None) -> tuple[str, str]:
    """Resolve the query data source for retrieval."""
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
    """Select CUDA when available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move only model inputs onto the selected device."""
    model_inputs: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in batch:
            model_inputs[key] = batch[key].to(device)
    return model_inputs


def _pool_word_vectors(sample_hidden: np.ndarray, word_ids: list[int], valid_length: int) -> dict[int, np.ndarray]:
    """Average subword vectors into one vector per original word."""
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


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize row vectors for cosine-style retrieval."""
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def _deduplicate_texts(texts: list[str]) -> list[str]:
    """Keep example order stable while removing exact duplicates."""
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)
    return unique_texts


def _coerce_text(value: Any) -> str | None:
    """Convert different input shapes into a single text string."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        stripped = " ".join(item.strip() for item in value if item.strip()).strip()
        return stripped or None
    if isinstance(value, dict):
        for key in ("text", "span_text", "entity", "entity_text", "phrase", "tokens"):
            if key in value:
                return _coerce_text(value[key])
    return None


def _coerce_tokens(value: Any) -> list[str]:
    """Convert raw text or token lists into the word sequence expected by the encoder."""
    if isinstance(value, list):
        tokens = [str(token).strip() for token in value]
        return [token for token in tokens if token]
    if isinstance(value, str):
        return [token for token in value.strip().split() if token]
    raise ValueError(f"Unsupported token source: {type(value)!r}")


def _normalize_support_examples(value: Any) -> list[str]:
    """Normalize rule support examples into a non-empty list of entity strings."""
    if isinstance(value, list):
        texts = [_coerce_text(item) for item in value]
        return _deduplicate_texts([text for text in texts if text])
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _parse_guideline_item(item: Any) -> tuple[str, str, list[str]] | None:
    """Parse one guideline rule entry from a supported JSON shape."""
    if isinstance(item, list) and len(item) >= 3:
        entity_type = _coerce_text(item[0])
        rule_text = _coerce_text(item[1])
        support_examples = _normalize_support_examples(item[2])
        if entity_type and rule_text and support_examples:
            return entity_type, rule_text, support_examples
        return None

    if not isinstance(item, dict):
        return None

    entity_type = next((_coerce_text(item.get(key)) for key in GUIDELINE_TYPE_KEYS if key in item), None)
    rule_text = next((_coerce_text(item.get(key)) for key in GUIDELINE_RULE_KEYS if key in item), None)
    support_examples = next(
        (_normalize_support_examples(item.get(key)) for key in GUIDELINE_INSTANCE_KEYS if key in item),
        [],
    )
    if entity_type and rule_text and support_examples:
        return entity_type, rule_text, support_examples
    return None


def load_guideline_specs(guideline_path: str | Path, max_rules: int | None = None) -> list[dict[str, Any]]:
    """Load guideline rules from JSON and flatten them into prototype specs."""
    payload = read_json(guideline_path)
    specs: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        # Native repository format: {"entity_type": {"rule_text": ["example_a", ...]}}
        if all(isinstance(value, dict) for value in payload.values()):
            for entity_type, rules in payload.items():
                for rule_text, support_examples in rules.items():
                    normalized_examples = _normalize_support_examples(support_examples)
                    if not normalized_examples:
                        continue
                    specs.append(
                        {
                            "entity_type": str(entity_type),
                            "rule_text": str(rule_text),
                            "support_examples": normalized_examples,
                        }
                    )
        else:
            for collection_key in ("guidelines", "rules", "items"):
                collection = payload.get(collection_key)
                if isinstance(collection, list):
                    for item in collection:
                        parsed = _parse_guideline_item(item)
                        if parsed is None:
                            continue
                        entity_type, rule_text, support_examples = parsed
                        specs.append(
                            {
                                "entity_type": entity_type,
                                "rule_text": rule_text,
                                "support_examples": support_examples,
                            }
                        )
                    break
    elif isinstance(payload, list):
        for item in payload:
            parsed = _parse_guideline_item(item)
            if parsed is None:
                continue
            entity_type, rule_text, support_examples = parsed
            specs.append(
                {
                    "entity_type": entity_type,
                    "rule_text": rule_text,
                    "support_examples": support_examples,
                }
            )
    else:
        raise ValueError("Guideline JSON must be a dict or list.")

    specs = [
        {
            "entity_type": spec["entity_type"].strip(),
            "rule_text": spec["rule_text"].strip(),
            "support_examples": _deduplicate_texts(spec["support_examples"]),
        }
        for spec in specs
        if spec["entity_type"].strip() and spec["rule_text"].strip() and spec["support_examples"]
    ]
    if max_rules is not None:
        specs = specs[: max(0, int(max_rules))]
    if not specs:
        raise ValueError(f"No valid guideline rules found in {Path(guideline_path).resolve()}.")
    return specs


def build_guideline_examples(prototype_specs: list[dict[str, Any]], guideline_path: str | Path) -> tuple[list[NERExample], dict[str, int]]:
    """Convert guideline entity instances into NERExample objects for encoding."""
    examples: list[NERExample] = []
    sample_to_prototype: dict[str, int] = {}
    source_path = str(Path(guideline_path).resolve())

    for prototype_index, spec in enumerate(prototype_specs):
        for example_index, support_example in enumerate(spec["support_examples"]):
            tokens = _coerce_tokens(support_example)
            if not tokens:
                continue
            sample_id = f"guideline-{prototype_index}-{example_index}"
            examples.append(
                NERExample(
                    sample_id=sample_id,
                    tokens=tokens,
                    ner_tags=["O"] * len(tokens),
                    split="guideline",
                    source_path=source_path,
                )
            )
            sample_to_prototype[sample_id] = prototype_index
    if not examples:
        raise ValueError("Guideline rules were parsed, but none contained encodable entity instances.")
    return examples, sample_to_prototype


def _load_text_jsonl_examples(path: str | Path, split: str, max_samples: int | None = None) -> list[NERExample]:
    """Read JSONL files that store raw text or tokenized inputs for retrieval queries."""
    file_path = Path(path)
    examples: list[NERExample] = []
    with file_path.open("r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)

            tokens_source = None
            if "tokens" in record:
                tokens_source = record["tokens"]
            elif "text" in record:
                tokens_source = record["text"]
            elif "sentence" in record:
                tokens_source = record["sentence"]

            if tokens_source is None:
                raise ValueError(
                    f"JSONL record at {file_path}:{idx + 1} must contain 'tokens', 'text', or 'sentence'."
                )

            tokens = _coerce_tokens(tokens_source)
            if not tokens:
                continue

            ner_tags = record.get("ner_tags")
            has_labels = isinstance(ner_tags, list) and len(ner_tags) == len(tokens)
            if not has_labels:
                ner_tags = ["O"] * len(tokens)

            examples.append(
                NERExample(
                    sample_id=str(record.get("sample_id", f"{split}-{idx}")),
                    tokens=tokens,
                    ner_tags=[str(tag) for tag in ner_tags],
                    split=str(record.get("split", split)),
                    source_path=str(file_path.resolve()),
                    has_labels=has_labels,
                )
            )
            if max_samples is not None and len(examples) >= max(0, int(max_samples)):
                break
    return examples


def _peek_first_jsonl_record(path: str | Path) -> dict[str, Any] | None:
    """Inspect the first non-empty JSONL record without loading the whole file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                return json.loads(stripped)
    return None


def load_query_examples(
    config: dict[str, Any],
    split: str,
    stage: str,
    input_path: str | None = None,
    max_samples: int | None = None,
) -> list[NERExample]:
    """Load retrieval queries from the configured split or an explicit input file."""
    data_path, input_format = _resolve_data_path(config, split=split, input_path=input_path)
    file_path = Path(data_path)
    normalized_split = normalize_split(split)
    normalized_stage = normalize_stage(stage)

    if normalized_split == "test":
        if normalized_stage == "test_infer":
            return load_test_text_only(
                file_path,
                input_format=input_format,
                max_samples=max_samples,
                stage=normalized_stage,
                split_name=normalized_split,
            )
        if normalized_stage == "final_eval":
            return load_test_with_labels_for_final_eval(
                file_path,
                input_format=input_format,
                max_samples=max_samples,
                stage=normalized_stage,
                split_name=normalized_split,
            )
        raise ValueError(
            f"Stage '{normalized_stage}' is not allowed to access test retrieval queries."
        )

    if file_path.suffix.lower() == ".jsonl":
        first_record = _peek_first_jsonl_record(file_path)
        if first_record is not None and any(key in first_record for key in ("tokens", "text", "sentence")):
            if normalized_split == "train":
                ensure_train_access(normalized_stage, operation="load_query_examples()", source_path=str(file_path))
                return _load_text_jsonl_examples(file_path, split=normalized_split, max_samples=max_samples)
            if normalized_split == "dev":
                ensure_dev_access(normalized_stage, operation="load_query_examples()", source_path=str(file_path))
                return _load_text_jsonl_examples(file_path, split=normalized_split, max_samples=max_samples)

    if normalized_split == "train":
        return load_train_data(
            file_path,
            input_format=input_format,
            max_samples=max_samples,
            stage=normalized_stage,
        )
    if normalized_split == "dev":
        return load_dev_data(
            file_path,
            input_format=input_format,
            max_samples=max_samples,
            stage=normalized_stage,
            split_name=split,
        )
    raise ValueError(f"Unsupported split '{split}' for retrieval query loading.")


def encode_examples(
    model: Any,
    tokenizer: Any,
    examples: list[NERExample],
    batch_size: int,
    max_length: int,
    hidden_state_layer: int,
    desc: str,
    progress_bar: Any | None = None,
) -> list[dict[str, Any]]:
    """Encode examples and return one pooled vector per visible word."""
    device = _get_device()
    model.to(device)
    model.eval()

    features = build_inference_features(examples, tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(features, batch_size=batch_size, shuffle=False, collate_fn=ExportCollator(tokenizer))

    encoded_records: list[dict[str, Any]] = []
    dataloader_iter = dataloader
    if progress_bar is None:
        dataloader_iter = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch in dataloader_iter:
            outputs = model(**_build_model_inputs(batch, device), output_hidden_states=True)
            hidden_states = outputs.hidden_states[hidden_state_layer].detach().cpu().numpy()
            attention_mask = batch["attention_mask"].detach().cpu()

            for batch_index, sample_id in enumerate(batch["sample_id"]):
                sequence_length = int(attention_mask[batch_index].sum().item())
                sample_hidden = hidden_states[batch_index, :sequence_length]
                sample_word_ids = batch["word_ids"][batch_index][:sequence_length]
                word_vectors = _pool_word_vectors(sample_hidden, sample_word_ids, sequence_length)

                encoded_records.append(
                    {
                        "sample_id": sample_id,
                        "split": batch["split"][batch_index],
                        "source_path": batch["source_path"][batch_index],
                        "tokens": list(batch["tokens"][batch_index]),
                        "ner_tags": list(batch["ner_tags"][batch_index]),
                        "has_labels": bool(batch["has_labels"][batch_index]),
                        "word_vectors": word_vectors,
                    }
                )
            if progress_bar is not None:
                progress_bar.update(len(batch["sample_id"]))
    return encoded_records


def build_guideline_prototypes(
    model: Any,
    tokenizer: Any,
    prototype_specs: list[dict[str, Any]],
    guideline_path: str | Path,
    batch_size: int,
    max_length: int,
    hidden_state_layer: int,
    output_dir: str | Path,
    checkpoint_path: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Encode guideline entity instances and pool them into per-rule prototypes."""
    examples, sample_to_prototype = build_guideline_examples(prototype_specs, guideline_path)
    encoded_records = encode_examples(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=batch_size,
        max_length=max_length,
        hidden_state_layer=hidden_state_layer,
        desc="Encoding guideline instances",
    )

    instance_vectors_by_prototype: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(prototype_specs))}
    for record in encoded_records:
        if not record["word_vectors"]:
            continue
        ordered_vectors = [record["word_vectors"][idx] for idx in sorted(record["word_vectors"].keys())]
        instance_vector = np.mean(np.stack(ordered_vectors, axis=0), axis=0).astype(np.float32)
        prototype_index = sample_to_prototype[record["sample_id"]]
        instance_vectors_by_prototype[prototype_index].append(instance_vector)

    prototype_vectors: list[np.ndarray] = []
    prototype_metadata: list[dict[str, Any]] = []
    for prototype_index, spec in enumerate(prototype_specs):
        instance_vectors = instance_vectors_by_prototype[prototype_index]
        if not instance_vectors:
            continue
        prototype_vector = np.mean(np.stack(instance_vectors, axis=0), axis=0).astype(np.float32)
        prototype_vectors.append(prototype_vector)
        prototype_metadata.append(
            {
                "prototype_index": len(prototype_metadata),
                "entity_type": spec["entity_type"],
                "rule_text": spec["rule_text"],
                "support_examples": list(spec["support_examples"]),
                "num_support_examples": len(spec["support_examples"]),
                "num_encoded_examples": len(instance_vectors),
                "checkpoint_path": checkpoint_path,
                "guideline_path": str(Path(guideline_path).resolve()),
                "pooling": {
                    "instance_pooling": "mean_over_words",
                    "prototype_pooling": "mean_over_instances",
                },
            }
        )

    if not prototype_vectors:
        raise ValueError("All guideline instances were skipped during prototype construction.")

    prototype_array = np.stack(prototype_vectors, axis=0).astype(np.float32)
    output_root = ensure_dir(output_dir)
    np.save(output_root / "vectors.npy", prototype_array)
    write_jsonl(output_root / "metadata.jsonl", prototype_metadata)
    summary = {
        "prototype_count": int(prototype_array.shape[0]),
        "hidden_size": int(prototype_array.shape[1]),
        "guideline_path": str(Path(guideline_path).resolve()),
        "checkpoint_path": checkpoint_path,
        "instance_count": len(examples),
        "encoded_instance_count": int(sum(item["num_encoded_examples"] for item in prototype_metadata)),
    }
    write_json(output_root / "summary.json", summary)
    return prototype_array, prototype_metadata, summary


def retrieve_guideline_prototypes(
    model: Any,
    tokenizer: Any,
    query_examples: list[NERExample],
    prototype_vectors: np.ndarray,
    prototype_metadata: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    hidden_state_layer: int,
    top_k: int,
    output_dir: str | Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Encode query sentences word by word and retrieve the closest guideline prototypes."""
    retrieval_records, summary = retrieve_guideline_records(
        model=model,
        tokenizer=tokenizer,
        query_examples=query_examples,
        prototype_vectors=prototype_vectors,
        prototype_metadata=prototype_metadata,
        batch_size=batch_size,
        max_length=max_length,
        hidden_state_layer=hidden_state_layer,
        top_k=top_k,
        checkpoint_path=checkpoint_path,
    )

    output_root = ensure_dir(output_dir)
    write_jsonl(output_root / "results.jsonl", retrieval_records)
    write_json(output_root / "summary.json", summary)
    return summary


def retrieve_guideline_records(
    model: Any,
    tokenizer: Any,
    query_examples: list[NERExample],
    prototype_vectors: np.ndarray,
    prototype_metadata: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    hidden_state_layer: int,
    top_k: int,
    checkpoint_path: str,
    progress_bar: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return sentence-level prototype retrieval records in memory without writing to disk."""
    encoded_records = encode_examples(
        model=model,
        tokenizer=tokenizer,
        examples=query_examples,
        batch_size=batch_size,
        max_length=max_length,
        hidden_state_layer=hidden_state_layer,
        desc="Encoding retrieval queries",
        progress_bar=progress_bar,
    )

    normalized_prototypes = _normalize_rows(prototype_vectors)
    retrieval_records: list[dict[str, Any]] = []
    total_visible_tokens = 0
    k = min(max(1, int(top_k)), len(prototype_metadata))

    for record in encoded_records:
        word_vectors = record["word_vectors"]
        visible_word_indices = sorted(word_vectors.keys())
        sentence_hits: dict[int, dict[str, Any]] = {}

        for word_index in visible_word_indices:
            query_vector = word_vectors[word_index].astype(np.float32, copy=False).reshape(1, -1)
            scores = (normalized_prototypes @ _normalize_rows(query_vector).reshape(-1)).astype(np.float32)
            token_text = record["tokens"][word_index]
            for prototype_index, score in enumerate(scores):
                score_value = float(score)
                if prototype_index not in sentence_hits:
                    metadata = prototype_metadata[prototype_index]
                    sentence_hits[prototype_index] = {
                        "score": score_value,
                        "prototype_index": int(prototype_index),
                        "entity_type": metadata["entity_type"],
                        "rule_text": metadata["rule_text"],
                        "support_examples": metadata["support_examples"],
                        "num_support_examples": metadata["num_support_examples"],
                        # Keep only the strongest matching word positions for each prototype.
                        "matched_tokens": [token_text],
                        "matched_word_indices": [word_index],
                        "best_word_index": word_index,
                    }
                    continue

                entry = sentence_hits[prototype_index]
                if score_value > entry["score"]:
                    entry["score"] = score_value
                    entry["best_word_index"] = word_index
                    entry["matched_tokens"] = [token_text]
                    entry["matched_word_indices"] = [word_index]
                elif np.isclose(score_value, entry["score"], rtol=1e-5, atol=1e-8):
                    if word_index not in entry["matched_word_indices"]:
                        entry["matched_word_indices"].append(word_index)
                    if token_text not in entry["matched_tokens"]:
                        entry["matched_tokens"].append(token_text)

        prototype_retrievals = sorted(sentence_hits.values(), key=lambda item: item["score"], reverse=True)[:k]
        for rank, hit in enumerate(prototype_retrievals, start=1):
            hit["rank"] = rank

        total_visible_tokens += len(visible_word_indices)
        retrieval_records.append(
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "source_path": record["source_path"],
                "checkpoint_path": checkpoint_path,
                "tokens": record["tokens"],
                "visible_tokens": [record["tokens"][idx] for idx in visible_word_indices],
                "truncated": len(visible_word_indices) < len(record["tokens"]),
                "prototype_retrievals": prototype_retrievals,
            }
        )
        if record["has_labels"]:
            retrieval_records[-1]["gold_tags"] = record["ner_tags"]

    summary = {
        "query_count": len(retrieval_records),
        "visible_token_count": total_visible_tokens,
        "top_k": int(top_k),
        "prototype_count": int(prototype_vectors.shape[0]),
        "checkpoint_path": checkpoint_path,
        "query_vector_type": "word",
        "prototype_vector_type": "guideline_rule",
    }
    return retrieval_records, summary

def load_saved_prototypes(prototype_dir: str | Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Load saved prototype vectors, metadata, and summary from disk."""
    prototype_root = Path(prototype_dir).resolve()
    vectors_path = prototype_root / "vectors.npy"
    metadata_path = prototype_root / "metadata.jsonl"
    summary_path = prototype_root / "summary.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Prototype vectors not found: {vectors_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Prototype metadata not found: {metadata_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Prototype summary not found: {summary_path}")

    prototype_vectors = np.load(vectors_path).astype(np.float32, copy=False)
    prototype_metadata = read_jsonl(metadata_path)
    prototype_summary = read_json(summary_path)
    return prototype_vectors, prototype_metadata, prototype_summary

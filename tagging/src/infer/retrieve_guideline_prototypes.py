"""Inference-time CLI for loading saved prototypes and retrieving them per token."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..models.deberta_token_classifier import load_checkpoint_model, load_tokenizer
from ..utils.config import load_config
from ..utils.io import write_json
from ..data.split_guard import normalize_split
from .guideline_retrieval import load_query_examples, load_saved_prototypes, retrieve_guideline_prototypes
from .prototype_paths import default_prototype_dir_from_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for online prototype retrieval."""
    parser = argparse.ArgumentParser(description="Load saved guideline prototypes and retrieve them for each token.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint-path", required=True, help="Fine-tuned checkpoint directory.")
    parser.add_argument(
        "--prototype-dir",
        default=None,
        help="Directory that contains saved prototype files. Defaults to GuideNER/prototypes/{model}-{dataset}-prototypes.",
    )
    parser.add_argument("--split", default="test", help="Named query split: train/validation/test.")
    parser.add_argument("--input-path", default=None, help="Optional explicit query file path.")
    parser.add_argument("--stage", default=None, help="Execution stage: train/dev. Test retrieval export is forbidden.")
    parser.add_argument("--output-dir", default=None, help="Directory for retrieval outputs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--max-query-samples", type=int, default=None, help="Optional cap for query examples.")
    parser.add_argument("--hidden-state-layer", type=int, default=None, help="Layer index to encode.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved guidelines per token.")
    return parser.parse_args()


def _default_stage_for_split(split: str) -> str:
    normalized_split = normalize_split(split)
    if normalized_split == "train":
        return "train"
    if normalized_split == "dev":
        return "dev"
    raise ValueError(
        "retrieve_guideline_prototypes.py must not export retrieval results for the test split."
    )


def main() -> None:
    """Run inference-time retrieval against previously saved guideline prototypes."""
    args = parse_args()
    stage = args.stage or _default_stage_for_split(args.split)
    config = load_config(
        args.config,
        overrides={
            "inference.batch_size": args.batch_size,
            "export.hidden_state_layer": args.hidden_state_layer,
        },
    )
    if normalize_split(args.split) == "test":
        raise ValueError(
            "Standalone retrieval export on the test split is forbidden. "
            "Use the final inference script instead."
        )

    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    prototype_dir = str(
        Path(args.prototype_dir).resolve() if args.prototype_dir else default_prototype_dir_from_config(config)
    )
    tokenizer = load_tokenizer(checkpoint_path)
    model = load_checkpoint_model(checkpoint_path, output_hidden_states=True)
    query_examples = load_query_examples(
        config,
        split=args.split,
        stage=stage,
        input_path=args.input_path,
        max_samples=args.max_query_samples,
    )
    prototype_vectors, prototype_metadata, prototype_summary = load_saved_prototypes(prototype_dir)

    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config["training"]["output_dir"]).resolve()
        / "guideline_retrieval"
        / "retrieval"
        / (args.split if args.input_path is None else "custom")
    )
    retrieval_summary = retrieve_guideline_prototypes(
        model=model,
        tokenizer=tokenizer,
        query_examples=query_examples,
        prototype_vectors=prototype_vectors,
        prototype_metadata=prototype_metadata,
        batch_size=int(config["inference"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        hidden_state_layer=int(config["export"]["hidden_state_layer"]),
        top_k=int(args.top_k),
        output_dir=output_root,
        checkpoint_path=checkpoint_path,
    )

    write_json(
        output_root / "retrieve_summary.json",
        {
            "checkpoint_path": checkpoint_path,
            "prototype_dir": prototype_dir,
            "prototype_summary": prototype_summary,
            "retrieval_summary": retrieval_summary,
            "query_split": args.split,
            "query_input_path": str(Path(args.input_path).resolve()) if args.input_path else None,
        },
    )


if __name__ == "__main__":
    main()

"""Offline CLI for building and saving guideline prototypes."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..models.deberta_token_classifier import load_checkpoint_model, load_tokenizer
from ..utils.config import load_config
from ..utils.io import write_json
from .guideline_retrieval import build_guideline_prototypes, load_guideline_specs
from .prototype_paths import default_prototype_dir_from_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline prototype construction."""
    parser = argparse.ArgumentParser(description="Build and save guideline prototypes from a guideline JSON file.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint-path", required=True, help="Fine-tuned checkpoint directory.")
    parser.add_argument("--guideline-path", required=True, help="Guideline JSON path.")
    parser.add_argument("--output-dir", default=None, help="Directory for saved prototype files.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--max-guideline-rules", type=int, default=None, help="Optional cap for guideline rules.")
    parser.add_argument("--hidden-state-layer", type=int, default=None, help="Layer index to encode.")
    return parser.parse_args()


def main() -> None:
    """Build and save guideline prototypes as an offline preprocessing step."""
    args = parse_args()
    config = load_config(
        args.config,
        overrides={
            "inference.batch_size": args.batch_size,
            "export.hidden_state_layer": args.hidden_state_layer,
        },
    )

    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    guideline_path = str(Path(args.guideline_path).resolve())
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else default_prototype_dir_from_config(config)
    )

    tokenizer = load_tokenizer(checkpoint_path)
    model = load_checkpoint_model(checkpoint_path, output_hidden_states=True)
    prototype_specs = load_guideline_specs(guideline_path, max_rules=args.max_guideline_rules)
    _, _, prototype_summary = build_guideline_prototypes(
        model=model,
        tokenizer=tokenizer,
        prototype_specs=prototype_specs,
        guideline_path=guideline_path,
        batch_size=int(config["inference"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        hidden_state_layer=int(config["export"]["hidden_state_layer"]),
        output_dir=output_root,
        checkpoint_path=checkpoint_path,
    )

    write_json(
        output_root / "build_summary.json",
        {
            "checkpoint_path": checkpoint_path,
            "guideline_path": guideline_path,
            "prototype_summary": prototype_summary,
        },
    )


if __name__ == "__main__":
    main()

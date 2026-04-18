import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from guide_dataset_io import load_label_schema, load_test_with_labels_for_final_eval
from tagging.src.infer.prototype_paths import infer_model_name_from_guideline_path


def normalize_entity_type(entity_type: str) -> str:
    """Normalize entity type names before metric comparison."""
    normalized = entity_type.strip().lower()
    if "geo" in normalized:
        return "geopolitical entity"
    return normalized


def normalize_label_pair(label: list) -> tuple[str, str] | None:
    """Normalize one [entity_text, entity_type] pair, or return None if malformed."""
    if not isinstance(label, list) or len(label) != 2:
        return None
    if not all(isinstance(item, str) for item in label):
        return None
    return label[0].strip().lower(), normalize_entity_type(label[1])


class Evaluate:
    """Entity-level precision / recall / F1 on exact span-text and type match."""

    def __init__(self) -> None:
        self.correct_preds = 0
        self.total_correct = 0
        self.total_preds = 0

    def update(self, label_preds: list, labels: list, allowed_types: set[str]) -> None:
        gold_labels = []
        for label in labels:
            normalized = normalize_label_pair(label)
            if normalized is not None:
                gold_labels.append(normalized)

        pred_labels = []
        for label in label_preds:
            normalized = normalize_label_pair(label)
            if normalized is None:
                continue
            if normalized[1] not in allowed_types:
                continue
            pred_labels.append(normalized)

        gold_pool = list(gold_labels)
        for pred_label in pred_labels:
            if pred_label in gold_pool:
                self.correct_preds += 1
                gold_pool.remove(pred_label)

        self.total_correct += len(gold_labels)
        self.total_preds += len(pred_labels)

    def evaluate(self) -> tuple[float, float, float]:
        precision = self.correct_preds / self.total_preds if self.total_preds else 0.0
        recall = self.correct_preds / self.total_correct if self.total_correct else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1


dataset_path_dict = {
    "conll2003": "./datasets/conll2003",
    "ace04": "./datasets/ace04",
    "ace05": "./datasets/ace05",
    "genia": "./datasets/genia",
}


def load_allowed_types(label_file: str) -> set[str]:
    """Load the allowed label vocabulary for prediction filtering."""
    with open(label_file, "r", encoding="utf8") as f:
        labels = json.loads(f.readline())
    return {normalize_entity_type(label_name) for label_name in labels.keys()}


def iter_jsonl(path: str) -> Iterable[dict]:
    """Yield JSONL records from disk."""
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def build_default_result_file(dataset_path: str, dataset_name: str, prototype_dir: str | None, model_name: str) -> str:
    """Build the default evaluation file path using the prototype summary's guideline source model."""
    resolved_prototype_dir = (
        Path(prototype_dir).resolve()
        if prototype_dir is not None
        else (Path("prototypes") / f"{model_name}-{dataset_name}-prototypes").resolve()
    )
    summary_path = resolved_prototype_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Could not resolve the default result file because prototype summary.json was not found at "
            f"{summary_path}. Provide --result_file or --prototype_dir."
        )

    with summary_path.open("r", encoding="utf8") as f:
        prototype_summary = json.load(f)

    guideline_path = prototype_summary.get("guideline_path")
    if not isinstance(guideline_path, str) or not guideline_path.strip():
        raise ValueError("Prototype summary must contain a non-empty 'guideline_path' for result naming.")

    guideline_model_name = infer_model_name_from_guideline_path(guideline_path)
    return os.path.join(dataset_path, f"{guideline_model_name}_word-level_{model_name}_result.jsonl")


def main():
    stage = "final_eval"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="conll2003", choices=["conll2003", "ace04", "ace05", "genia"])
    parser.add_argument("--model_name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--prototype_dir", default=None)
    parser.add_argument("--result_file", default=None)
    args = parser.parse_args()

    dataset_path = dataset_path_dict[args.dataset_name]
    eval_file = (
        args.result_file
        if args.result_file
        else build_default_result_file(dataset_path, args.dataset_name, args.prototype_dir, args.model_name)
    )
    allowed_types = {
        normalize_entity_type(label_name)
        for label_name in load_label_schema(dataset_path).keys()
    }
    gold_records = load_test_with_labels_for_final_eval(dataset_path, stage=stage)
    prediction_records = list(iter_jsonl(eval_file))
    if len(prediction_records) != len(gold_records):
        raise ValueError(
            f"Prediction count {len(prediction_records)} does not match gold test count {len(gold_records)}."
        )
    evaluator = Evaluate()
    invalid_count = 0
    success_count = 0

    for idx, (record, gold_record) in enumerate(zip(prediction_records, gold_records), start=1):
        prediction_text = record.get("text")
        if not isinstance(prediction_text, str):
            raise ValueError(f"Prediction record {idx} must contain a string 'text' field.")
        if prediction_text != gold_record["text"]:
            raise ValueError(f"Prediction text at record {idx} does not match the final-eval test set order.")

        if record.get("status") != "success":
            invalid_count += 1
            continue
        success_count += 1
        evaluator.update(
            record.get("labels", []),
            gold_record["entity_labels"],
            allowed_types,
        )

    precision, recall, f1 = evaluator.evaluate()
    print(f"eval_file: {eval_file}")
    print(f"success_count: {success_count}")
    print(f"invalid_count: {invalid_count}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")


if __name__ == "__main__":
    main()

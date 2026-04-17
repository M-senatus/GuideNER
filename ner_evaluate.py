import argparse
import json
import os
from typing import Iterable


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="conll2003", choices=["conll2003", "ace04", "ace05", "genia"])
    parser.add_argument("--model_name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--result_file", default=None)
    args = parser.parse_args()

    dataset_path = dataset_path_dict[args.dataset_name]
    eval_file = (
        args.result_file
        if args.result_file
        else os.path.join(dataset_path, f"{args.model_name}_withrule_retrieval_result_detail.jsonl")
    )
    label_file = os.path.join(dataset_path, "labels.jsonl")

    allowed_types = load_allowed_types(label_file)
    evaluator = Evaluate()
    invalid_count = 0
    success_count = 0

    for record in iter_jsonl(eval_file):
        if record.get("status") != "success":
            invalid_count += 1
            continue
        success_count += 1
        evaluator.update(
            record.get("predicted_labels", []),
            record.get("labels", []),
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

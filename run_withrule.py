import argparse
import ast
import json
import os
import re
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from EasyChatTemplating.util_tools import convert_userprompt_transformers, skip_special_tokens_transformers
from tagging.src.infer.guideline_retrieval import (
    load_query_examples,
    load_saved_prototypes,
    retrieve_guideline_records,
)
from tagging.src.infer.prototype_paths import (
    infer_model_name_from_guideline_path,
    prototype_dir_from_model_and_dataset,
)
from tagging.src.models.deberta_token_classifier import (
    load_checkpoint_model,
    load_tokenizer as load_ner_tokenizer,
)
from tagging.src.data.schemas import NERExample
from tagging.src.utils.config import load_config
from guide_dataset_io import load_test_text_only


label_pattern = r"\[\[(.*?)\]\]"
MODEL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "model"))
DEFAULT_MODEL_NAME = "Llama-3.1-8B-Instruct"
DEFAULT_TAGGING_CONFIG = os.path.join("tagging", "configs", "deberta_ner_conll2003.json")
model_path_dict = {
    "bge-m3": os.path.join(MODEL_ROOT, "bge-m3"),
    "Llama-3.1-8B-Instruct": os.path.join(MODEL_ROOT, "Llama-3.1-8B-Instruct"),
    "Ministral-3-8B-Instruct-2512": os.path.join(MODEL_ROOT, "Ministral-3-8B-Instruct-2512"),
    "Qwen2.5-7B-Instruct": os.path.join(MODEL_ROOT, "Qwen2.5-7B-Instruct"),
}
dataset_path_dict = {
    "conll2003": "./datasets/conll2003",
    "ace04": "./datasets/ace04",
    "ace05": "./datasets/ace05",
    "genia": "./datasets/genia",
}


conll2003_rule_prompt = """Task: Please identify Person, Organization, Location and Miscellaneous Entity from the given text and rules. 
The rules provide an entity category followed by a list of patterns that match that category.

Rules:
{Rules}
Please note: Patterns not included in the above are not entities.

Examples:
Input Text: EU rejects German call to boycott British lamb.
Given the Input Text and Rules, only classify text as an entity if it matches a pattern; otherwise, it should not be classified as an entity. 
The Output is: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]
Input Text: S&P = DENOMS ( K ) 1-10-100 SALE LIMITS US / UK / CA
Given the Input Text and Rules, only classify text as an entity if it matches a pattern; otherwise, it should not be classified as an entity. 
The Output is: [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"], ["Zhirinovsky", "person"]]
Input Text: -- E. Auchard , Wall Street bureau , 212-859-1736
Given the Input Text and Rules, only classify text as an entity if it matches a pattern; otherwise, it should not be classified as an entity. 
The Output is: [["E. Auchard", "person"], ["Wall Street bureau", "organization"]]

Identify Entities for: 
Input Text: {input_text}
Given the Input Text and Rules, only classify text as an entity if it matches a pattern; otherwise, it should not be classified as an entity. 
The Output is:
"""


def build_sentence_guideline_summary(prototype_retrievals: list[dict], max_guidelines: int | None = None) -> list[dict]:
    """Normalize sentence-level prototype retrievals for prompt construction."""
    summary = [
        {
            "entity_type": str(item["entity_type"]).lower(),
            "rule_text": str(item["rule_text"]),
            "best_score": float(item["score"]),
            "support_examples": list(item.get("support_examples", [])),
            "matched_tokens": list(item.get("matched_tokens", [])),
            "matched_word_indices": list(item.get("matched_word_indices", [])),
            "best_word_index": item.get("best_word_index"),
        }
        for item in prototype_retrievals
    ]
    if max_guidelines is not None:
        summary = summary[: max(0, int(max_guidelines))]
    return summary


def format_guidelines_for_prompt(guideline_summary: list[dict]) -> str:
    """Render merged retrieved guidelines in the original category -> rule list format."""
    if not guideline_summary:
        return ""

    grouped_rules: dict[str, list[str]] = {}
    for item in guideline_summary:
        entity_type = item["entity_type"]
        grouped_rules.setdefault(entity_type, [])
        if item["rule_text"] not in grouped_rules[entity_type]:
            grouped_rules[entity_type].append(item["rule_text"])

    lines = []
    for entity_type, rules in grouped_rules.items():
        lines.append(f"{entity_type.capitalize()}: {rules}")
    return "\n".join(lines)


def parse_prediction(text: str) -> tuple[str, list]:
    """Parse a model prediction into status plus structured labels."""
    result = re.search(label_pattern, text, re.DOTALL)
    if result is None:
        return "none_wrong", []

    try:
        parsed = ast.literal_eval(result.group())
    except (SyntaxError, ValueError):
        return "eval_wrong", []

    if not isinstance(parsed, list):
        return "eval_wrong", []
    return "success", parsed


def build_default_result_file(dataset_path: str, prototype_summary: dict, entity_model_name: str) -> str:
    """Build the default result filename from the guideline-source model and inference LLM."""
    guideline_path = prototype_summary.get("guideline_path")
    if not isinstance(guideline_path, str) or not guideline_path.strip():
        raise ValueError("Prototype summary must contain a non-empty 'guideline_path' for result naming.")

    guideline_model_name = infer_model_name_from_guideline_path(guideline_path)
    return os.path.join(dataset_path, f"{guideline_model_name}_word-level_{entity_model_name}_result.jsonl")


def predict_batch(outputs, tokenizer, fw, batch_records):
    """Write one batch of text-only final test predictions."""
    for output, record in zip(outputs, batch_records):
        generated_text = skip_special_tokens_transformers(tokenizer, output.outputs[0].text)
        status, labels = parse_prediction(generated_text)

        result_dict = {
            "text": record["text"],
            "labels": labels,
            "status": status,
            "guideline": record["guideline"],
        }

        fw.write(json.dumps(result_dict, ensure_ascii=False))
        fw.write("\n")
        fw.flush()


def run_llm_batch(messages, llm, sampling_params):
    """Run one LLM batch without showing vLLM's internal progress bar."""
    return llm.generate(messages, sampling_params, use_tqdm=False)


def resolve_runtime_paths(args):
    """Resolve model, checkpoint, and prototype paths for inference-time utilities."""
    model_path = model_path_dict[args.model_name]
    dataset_path = dataset_path_dict[args.dataset_name]
    tagging_config = load_config(args.tagging_config)
    ner_checkpoint_path = (
        str(Path(args.ner_checkpoint_path).resolve())
        if args.ner_checkpoint_path
        else str((Path(tagging_config["training"]["output_dir"]) / "checkpoint-best").resolve())
    )
    prototype_dir = (
        str(Path(args.prototype_dir).resolve())
        if args.prototype_dir
        else str(
            prototype_dir_from_model_and_dataset(
                tagging_config,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
            ).resolve()
        )
    )
    return model_path, dataset_path, tagging_config, ner_checkpoint_path, prototype_dir


def build_llm_runtime(args, model_path):
    """Initialize the LLM, tokenizer, and decoding parameters."""
    llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        max_model_len=32768,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    return llm_tokenizer, llm, sampling_params


def run_single_step_test(input_text: str, args) -> str:
    """Run one ad-hoc text example through retrieval plus LLM generation."""
    normalized_text = input_text.strip()
    if not normalized_text:
        raise ValueError("single-step test input_text must not be empty.")

    model_path, _, tagging_config, ner_checkpoint_path, prototype_dir = resolve_runtime_paths(args)
    llm_tokenizer, llm, sampling_params = build_llm_runtime(args, model_path)

    ner_tokenizer = load_ner_tokenizer(ner_checkpoint_path)
    ner_model = load_checkpoint_model(ner_checkpoint_path, output_hidden_states=True)
    prototype_vectors, prototype_metadata, _ = load_saved_prototypes(prototype_dir)

    # Keep the ad-hoc debug path text-only and unlabeled.
    query_example = NERExample(
        sample_id="single-step-input",
        tokens=normalized_text.split(),
        ner_tags=["O"] * len(normalized_text.split()),
        split="adhoc",
        source_path="<single_test_input>",
        has_labels=False,
    )
    retrieval_records, _ = retrieve_guideline_records(
        model=ner_model,
        tokenizer=ner_tokenizer,
        query_examples=[query_example],
        prototype_vectors=prototype_vectors,
        prototype_metadata=prototype_metadata,
        batch_size=1,
        max_length=int(tagging_config["model"]["max_length"]),
        hidden_state_layer=int(tagging_config["export"]["hidden_state_layer"]),
        top_k=int(args.retrieval_top_k),
        checkpoint_path=ner_checkpoint_path,
    )

    guideline_summary = build_sentence_guideline_summary(
        retrieval_records[0]["prototype_retrievals"],
        max_guidelines=args.retrieval_top_k,
    )
    prompt_guidelines = format_guidelines_for_prompt(guideline_summary)
    task_prompt = eval(f"{args.dataset_name}_rule_prompt")
    prompt_predict = task_prompt.format(Rules=prompt_guidelines, input_text=normalized_text)
    message = convert_userprompt_transformers(llm_tokenizer, prompt_predict, add_generation_prompt=True)
    outputs = run_llm_batch([message], llm, sampling_params)
    return skip_special_tokens_transformers(llm_tokenizer, outputs[0].outputs[0].text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="conll2003", choices=["conll2003", "ace04", "ace05", "genia"])
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, choices=list(model_path_dict.keys()))
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_tokens", default=1024, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--retrieval_top_k",
        default=10,
        type=int,
        help="Number of retrieved guideline prototypes to keep and include in the prompt.",
    )
    parser.add_argument("--tagging_config", default=DEFAULT_TAGGING_CONFIG)
    parser.add_argument("--ner_checkpoint_path", default=None)
    parser.add_argument("--prototype_dir", default=None)
    parser.add_argument("--result_file", default=None)
    parser.add_argument(
        "--single_test_input_text",
        default=None,
        help="Run one ad-hoc input_text through retrieval plus LLM generation and print the raw model output.",
    )
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main(args=None):
    stage = "test_infer"
    args = parse_args() if args is None else args
    model_path, dataset_path, tagging_config, ner_checkpoint_path, prototype_dir = resolve_runtime_paths(args)
    test_file = os.path.join(dataset_path, "test.jsonl")

    llm_tokenizer, llm, sampling_params = build_llm_runtime(args, model_path)

    ner_tokenizer = load_ner_tokenizer(ner_checkpoint_path)
    ner_model = load_checkpoint_model(ner_checkpoint_path, output_hidden_states=True)
    prototype_vectors, prototype_metadata, prototype_summary = load_saved_prototypes(prototype_dir)
    query_examples = load_query_examples(
        tagging_config,
        split="test",
        stage=stage,
        input_path=test_file,
        max_samples=None,
    )
    test_records = load_test_text_only(dataset_path, stage=stage)

    result_file_name = (
        args.result_file
        if args.result_file
        else build_default_result_file(dataset_path, prototype_summary, args.model_name)
    )
    file_mode = "a" if args.append else "w"
    fw = open(result_file_name, file_mode, encoding="utf8")

    task_prompt = eval(f"{args.dataset_name}_rule_prompt")
    messages = []
    batch_records = []
    overall_progress = tqdm(
        total=len(query_examples) + len(test_records),
        desc="Overall progress | retrieve + infer",
        unit="sample",
        dynamic_ncols=True,
    )

    try:
        retrieval_records, _ = retrieve_guideline_records(
            model=ner_model,
            tokenizer=ner_tokenizer,
            query_examples=query_examples,
            prototype_vectors=prototype_vectors,
            prototype_metadata=prototype_metadata,
            batch_size=int(tagging_config["inference"]["batch_size"]),
            max_length=int(tagging_config["model"]["max_length"]),
            hidden_state_layer=int(tagging_config["export"]["hidden_state_layer"]),
            top_k=int(args.retrieval_top_k),
            checkpoint_path=ner_checkpoint_path,
            progress_bar=overall_progress,
        )

        test_records = load_test_text_only(dataset_path, stage=stage)
        if len(test_records) != len(retrieval_records):
            raise ValueError(
                f"Test sample count {len(test_records)} does not match retrieval record count {len(retrieval_records)}."
            )

        for test_record, retrieval_record in zip(test_records, retrieval_records):
            text = test_record["text"]
            guideline_summary = build_sentence_guideline_summary(
                retrieval_record["prototype_retrievals"],
                max_guidelines=args.retrieval_top_k,
            )
            prompt_guidelines = format_guidelines_for_prompt(guideline_summary)
            prompt_predict = task_prompt.format(Rules=prompt_guidelines, input_text=text)
            message = convert_userprompt_transformers(llm_tokenizer, prompt_predict, add_generation_prompt=True)

            batch_records.append(
                {
                    "text": text,
                    "guideline": prompt_guidelines,
                }
            )
            messages.append(message)
            overall_progress.update(1)

            if len(messages) >= args.batch_size:
                outputs = run_llm_batch(messages, llm, sampling_params)
                predict_batch(outputs, llm_tokenizer, fw, batch_records)
                messages = []
                batch_records = []

        if messages:
            outputs = run_llm_batch(messages, llm, sampling_params)
            predict_batch(outputs, llm_tokenizer, fw, batch_records)
    finally:
        overall_progress.close()
        fw.close()


if __name__ == "__main__":
    cli_args = parse_args()


    def single_step_test(input_text: str) -> str:
        """Debug one input_text from the command line and print the raw model output."""
        generated_text = run_single_step_test(input_text, cli_args)
        print(generated_text)
        return generated_text


    if cli_args.single_test_input_text is not None:
        single_step_test(cli_args.single_test_input_text)
    else:
        main(cli_args)

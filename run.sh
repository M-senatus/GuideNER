#!/usr/bin/env bash
set -euo pipefail

# Run one repository-root pipeline stage at a time:
# 1) summarize rules from the training set
# 4) run text-only test inference with retrieved guidelines
# 4b) run one ad-hoc single-step test example
# 5) compute final test metrics from the saved predictions
#
# The inference stage depends on the fine-tuned NER checkpoint and guideline
# prototypes produced by README steps 2 and 3. Override defaults with
# environment variables when needed, for example:
# CUDA_DEVICES=1 MODEL_NAME=Qwen2.5-7B-Instruct DATASET_NAME=conll2003 bash run.sh infer

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-conll2003}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-10}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
RESULT_FILE="${RESULT_FILE:-}"
SAVE_RAW_OUTPUT="${SAVE_RAW_OUTPUT:-0}"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-prototypes/${MODEL_NAME}-${DATASET_NAME}-prototypes}"

usage() {
  cat <<EOF
Usage:
  bash run.sh summary [extra args...]
  bash run.sh infer [extra args...]
  bash run.sh single-test [extra args...]
  bash run.sh evaluate [extra args...]
  bash run.sh help

This script runs one repository-root stage per invocation:
  summary   Generate summary rules from the training split
  infer     Run full text-only test inference with retrieved guidelines
  single-test
            Run one ad-hoc input_text through retrieval plus LLM generation
  evaluate  Compute final metrics from saved predictions

Environment overrides:
  PYTHON_BIN     Python executable (default: python)
  MODEL_NAME     LLM name (default: Llama-3.1-8B-Instruct)
  DATASET_NAME   Dataset name (default: conll2003)
  TEMPERATURE    Generation temperature (default: 0)
  TOP_P          Generation top-p (default: 1)
  CUDA_DEVICES   CUDA device ids (default: 0)
  RESULT_FILE    Output JSONL path for final predictions (default: inferred by Python)
  SAVE_RAW_OUTPUT
                 Save companion raw LLM output JSONL during infer (default: 1)
  PROTOTYPE_DIR  Retrieved-guideline prototype directory

Examples:
  bash run.sh summary
  MODEL_NAME=Qwen2.5-7B-Instruct bash run.sh summary
  DATASET_NAME=ace05 CUDA_DEVICES=1 bash run.sh infer
  SAVE_RAW_OUTPUT=0 bash run.sh infer
  bash run.sh single-test --single_test_input_text "EU rejects German call to boycott British lamb."
  RESULT_FILE=datasets/conll2003/custom_result.jsonl bash run.sh evaluate
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory not found: $1" >&2
    exit 1
  fi
}

should_save_raw_output() {
  case "${SAVE_RAW_OUTPUT,,}" in
    1|true|yes|on)
      return 0
      ;;
    0|false|no|off)
      return 1
      ;;
    *)
      echo "Invalid SAVE_RAW_OUTPUT value: $SAVE_RAW_OUTPUT" >&2
      echo "Use one of: 1, 0, true, false, yes, no, on, off" >&2
      exit 1
      ;;
  esac
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export PYTHONUNBUFFERED=1

case "$COMMAND" in
  summary)
    "$PYTHON_BIN" -u rule_summary.py \
      --dataset_name "$DATASET_NAME" \
      --model_name "$MODEL_NAME" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      "$@"
    ;;

  infer)
    require_file "tagging/configs/deberta_ner_${DATASET_NAME}.json"
    require_dir "../model/deberta-v3-base/deberta_ner_${DATASET_NAME}/checkpoint-best"
    require_dir "$PROTOTYPE_DIR"

    infer_args=(
      --dataset_name "$DATASET_NAME"
      --model_name "$MODEL_NAME"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --retrieval_top_k "$RETRIEVAL_TOP_K"
      --tagging_config "tagging/configs/deberta_ner_${DATASET_NAME}.json"
      --ner_checkpoint_path "../model/deberta-v3-base/deberta_ner_${DATASET_NAME}/checkpoint-best"
      --prototype_dir "$PROTOTYPE_DIR"
    )
    if should_save_raw_output; then
      infer_args+=(--save_raw_output)
    fi
    if [[ -n "$RESULT_FILE" ]]; then
      infer_args+=(--result_file "$RESULT_FILE")
    fi

    "$PYTHON_BIN" -u run_withrule.py "${infer_args[@]}" "$@"
    ;;

  single-test)
    require_file "tagging/configs/deberta_ner_${DATASET_NAME}.json"
    require_dir "../model/deberta-v3-base/deberta_ner_${DATASET_NAME}/checkpoint-best"
    require_dir "$PROTOTYPE_DIR"

    "$PYTHON_BIN" -u run_withrule.py \
      --dataset_name "$DATASET_NAME" \
      --model_name "$MODEL_NAME" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --retrieval_top_k "$RETRIEVAL_TOP_K" \
      --tagging_config "tagging/configs/deberta_ner_${DATASET_NAME}.json" \
      --ner_checkpoint_path "../model/deberta-v3-base/deberta_ner_${DATASET_NAME}/checkpoint-best" \
      --prototype_dir "$PROTOTYPE_DIR" \
      "$@"
    ;;

  evaluate)
    evaluate_args=(
      --dataset_name "$DATASET_NAME"
      --model_name "$MODEL_NAME"
      --prototype_dir "$PROTOTYPE_DIR"
    )
    if [[ -n "$RESULT_FILE" ]]; then
      evaluate_args+=(--result_file "$RESULT_FILE")
    fi

    "$PYTHON_BIN" ner_evaluate.py "${evaluate_args[@]}" "$@"
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    echo "Unknown command: $COMMAND" >&2
    usage
    exit 1
    ;;
esac

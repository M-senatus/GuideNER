#!/usr/bin/env bash
set -euo pipefail

# Run the README pipeline steps that live at the repository root:
# 1) summarize rules from the training set
# 4) run text-only test inference with retrieved guidelines
# 5) compute final test metrics from the saved predictions
#
# Step 4 depends on the fine-tuned NER checkpoint and guideline prototypes
# produced by README steps 2 and 3. Override defaults with environment
# variables when needed, for example:
# CUDA_DEVICES=1 MODEL_NAME=Qwen2.5-7B-Instruct DATASET_NAME=conll2003 bash run.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-conll2003}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
RESULT_FILE="${RESULT_FILE:-datasets/${DATASET_NAME}/${MODEL_NAME}_withrule_retrieval_result_detail.jsonl}"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-prototypes/${MODEL_NAME}-${DATASET_NAME}-prototypes}"

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

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -u rule_summary.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P"

require_file "tagging/configs/deberta_ner_${DATASET_NAME}.json"
require_dir "../model/deberta-v3-base/deberta_ner_${DATASET_NAME}/checkpoint-best"
require_dir "$PROTOTYPE_DIR"

"$PYTHON_BIN" -u run_withrule.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --tagging_config "tagging/configs/deberta_ner_${DATASET_NAME}.json" \
  --ner_checkpoint_path "../model/deberta-v3-base/deberta_ner_${DATASET_NAME}/checkpoint-best" \
  --prototype_dir "$PROTOTYPE_DIR" \
  --result_file "$RESULT_FILE"

"$PYTHON_BIN" ner_evaluate.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --result_file "$RESULT_FILE"

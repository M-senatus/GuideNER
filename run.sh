#!/usr/bin/env bash
set -euo pipefail

# Run rule_summary.py with vLLM backend.
# Override defaults with environment variables when needed, for example:
# CUDA_VISIBLE_DEVICES=1 MODEL_NAME=Qwen2.5-7B-Instruct DATASET_NAME=conll2003 bash run.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU_DEVICE="${GPU_DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-conll2003}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

python rule_summary.py \
  --dataset_name "$DATASET_NAME" \
  --model_name "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P"

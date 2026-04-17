#!/usr/bin/env bash
set -euo pipefail

TAGGING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-$TAGGING_DIR/configs/deberta_ner_conll2003.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$TAGGING_DIR/outputs/deberta_ner_conll2003/checkpoint-best}"
SPLIT="${SPLIT:-test}"
VECTOR_TYPE="${VECTOR_TYPE:-word}"
SPAN_FILE="${SPAN_FILE:-}"

usage() {
  cat <<EOF
Usage:
  bash tagging/run.sh train [extra args...]
  bash tagging/run.sh eval [extra args...]
  bash tagging/run.sh predict [extra args...]
  bash tagging/run.sh export [token|word|span] [extra args...]
  bash tagging/run.sh smoke

Environment overrides:
  PYTHON_BIN       Python executable (default: python)
  CONFIG_PATH      Config file path
  CHECKPOINT_PATH  Fine-tuned checkpoint path
  SPLIT            Data split for eval/predict/export (default: test)
  VECTOR_TYPE      Default vector type for export (default: word)
  SPAN_FILE        Optional JSONL span file for span export

Examples:
  bash tagging/run.sh train
  bash tagging/run.sh eval
  bash tagging/run.sh predict
  bash tagging/run.sh export word
  SPLIT=validation bash tagging/run.sh eval
  CHECKPOINT_PATH="$TAGGING_DIR/outputs/deberta_ner_conll2003/checkpoint-best" bash tagging/run.sh export span
  bash tagging/run.sh smoke
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
  train)
    export PYTHONUNBUFFERED=1
    "$PYTHON_BIN" "$TAGGING_DIR/scripts/train_ner.py" \
      --config "$CONFIG_PATH" \
      "$@"
    ;;

  eval)
    export PYTHONUNBUFFERED=1
    "$PYTHON_BIN" "$TAGGING_DIR/scripts/eval_ner.py" \
      --config "$CONFIG_PATH" \
      --checkpoint-path "$CHECKPOINT_PATH" \
      --split "$SPLIT" \
      "$@"
    ;;

  predict)
    export PYTHONUNBUFFERED=1
    "$PYTHON_BIN" "$TAGGING_DIR/scripts/predict_ner.py" \
      --config "$CONFIG_PATH" \
      --checkpoint-path "$CHECKPOINT_PATH" \
      --split "$SPLIT" \
      "$@"
    ;;

  export)
    if [[ $# -gt 0 && "$1" != --* ]]; then
      VECTOR_TYPE="$1"
      shift
    fi

    if [[ "$VECTOR_TYPE" != "token" && "$VECTOR_TYPE" != "word" && "$VECTOR_TYPE" != "span" ]]; then
      echo "Unsupported vector type: $VECTOR_TYPE" >&2
      exit 1
    fi

    export PYTHONUNBUFFERED=1
    CMD=(
      "$PYTHON_BIN" "$TAGGING_DIR/scripts/export_vectors.py"
      --config "$CONFIG_PATH"
      --checkpoint-path "$CHECKPOINT_PATH"
      --split "$SPLIT"
      --vector-type "$VECTOR_TYPE"
    )

    if [[ -n "$SPAN_FILE" ]]; then
      CMD+=(--span-file "$SPAN_FILE")
    fi

    "${CMD[@]}" "$@"
    ;;

  smoke)
    export PYTHONUNBUFFERED=1
    "$PYTHON_BIN" "$TAGGING_DIR/scripts/train_ner.py" \
      --config "$CONFIG_PATH" \
      --max-train-samples 64 \
      --max-validation-samples 32 \
      --max-test-samples 32 \
      --num-train-epochs 1 \
      "$@"
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

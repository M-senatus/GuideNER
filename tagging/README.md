# DeBERTa NER Tagging

This directory contains a minimal, reproducible Hugging Face NER pipeline for:

- fine-tuning `microsoft/deberta-v3-base` on BIO-style NER
- evaluating with seqeval span-level metrics
- exporting token, word, and span vectors for downstream retrieval

## Install

Create an environment with at least:

```bash
pip install -r tagging/requirements.txt
```

## Data

The default config reads the existing CoNLL-style files in the repository:

- `datasets/conll2003/train.txt`
- `datasets/conll2003/valid.txt`
- `datasets/conll2003/test.txt`

The pipeline also supports JSONL input where each record contains:

```json
{"tokens": ["EU", "rejects"], "ner_tags": ["B-ORG", "O"]}
```

## Run Training

From the repository root:

```bash
bash tagging/run.sh train
```

Artifacts are written under:

- `tagging/outputs/deberta_ner_conll2003/checkpoints`
- `tagging/outputs/deberta_ner_conll2003/checkpoint-best`
- `tagging/outputs/deberta_ner_conll2003/artifacts`

## Run Evaluation

```bash
bash tagging/run.sh eval
```

## Run Prediction

```bash
bash tagging/run.sh predict
```

## Export Hidden States

Word-level export:

```bash
bash tagging/run.sh export word
```

Span-level export with gold entity spans:

```bash
bash tagging/run.sh export span
```

## Minimal Smoke-Test Example

For a quick smoke test, you can temporarily set small sample caps in the config:

- `data.max_train_samples`
- `data.max_validation_samples`
- `data.max_test_samples`
- `export.max_samples`

This keeps the pipeline cheap to debug before full training.

```bash
bash tagging/run.sh smoke
```

## Retrieval Interface Notes

The export step writes:

- `vectors.npy`
- `metadata.jsonl`
- `summary.json`

`metadata.jsonl` includes enough fields for later guideline retrieval and entity-level retrieval, such as:

- `sample_id`
- `split`
- `vector_type`
- `word_index`
- `span_start`
- `span_end`
- `span_text`
- `entity_type`
- `source_path`
- `checkpoint_path`

## Useful Overrides

You can override common settings with environment variables:

```bash
SPLIT=validation bash tagging/run.sh eval
CHECKPOINT_PATH=tagging/outputs/deberta_ner_conll2003/checkpoint-best bash tagging/run.sh export span
PYTHON_BIN=python bash tagging/run.sh train
```

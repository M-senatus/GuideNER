# DeBERTa NER Tagging

This directory contains a minimal, reproducible Hugging Face NER pipeline for:

- fine-tuning a local `deberta-v3-base` checkpoint on BIO-style NER
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

Default local model and large artifacts:

- base model: `../model/deberta-v3-base/deberta-v3-base`
- fine-tuned checkpoints and exported vectors: `../model/deberta-v3-base/deberta_ner_conll2003`

## Run Training

From the repository root:

```bash
bash tagging/run.sh train
```

Artifacts are written under:

- `../model/deberta-v3-base/deberta_ner_conll2003/checkpoints`
- `../model/deberta-v3-base/deberta_ner_conll2003/checkpoint-best`
- `../model/deberta-v3-base/deberta_ner_conll2003/artifacts`

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

## Build Guideline Prototypes

This is the offline stage. It:

1. reads guideline JSON with entity types, rule text, and entity instances
2. encodes each entity instance with the fine-tuned DeBERTa checkpoint
3. pools instance vectors into one prototype per rule
4. saves prototype vectors plus metadata

Example:

```bash
bash tagging/run.sh guideline-build \
  --guideline-path datasets/conll2003/Qwen2.5-7B-Instruct_summaryrules.json
```

By default outputs are written under `../prototypes/{model}-{dataset}-prototypes/`, for example
`../prototypes/deberta-v3-base-conll2003-prototypes/`, with:

- `vectors.npy`
- `metadata.jsonl`
- `summary.json`
- `build_summary.json`

## Retrieve Guideline Prototypes

This is the inference stage. It:

1. loads saved prototype vectors and metadata
2. encodes each token in the query split at the word level
3. retrieves top-k guideline prototypes for every token

Example:

```bash
bash tagging/run.sh guideline-retrieve \
  --prototype-dir ../../prototypes/deberta-v3-base-conll2003-prototypes \
  --top-k 5
```

Outputs are written under `.../guideline_retrieval/retrieval/<split>/` with:

- `results.jsonl`
- `summary.json`
- `retrieve_summary.json`

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
CHECKPOINT_PATH=../model/deberta-v3-base/deberta_ner_conll2003/checkpoint-best bash tagging/run.sh export span
PYTHON_BIN=python bash tagging/run.sh train
```

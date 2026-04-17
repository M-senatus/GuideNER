# GuideNER: Annotation Guidelines Are Better than Examples for In-Context Named Entity Recognition
The pytorch implementation of "GuideNER: Annotation Guidelines Are Better than Examples for In-Context Named Entity Recognition " (AAAI 2025).

The framework of GuideNER is shown in the following figure:

![framework](image.png)

## Enviroment
We recommend the following actions to create the environment:

```bash
conda create -n  GuideNER python==3.9.19
conda activate GuideNER
pip install jinja2==3.1.4
pip install transformers==4.43.3
pip install vllm==0.5.3.post1
pip install tokenizers==0.19.1
pip install -r tagging/requirements.txt
```

The repository assumes local model checkpoints are stored under `../model/`. For the example below, prepare at least:

- `../model/Llama-3.1-8B-Instruct`
- `../model/deberta-v3-base/deberta-v3-base`

## Datasets
Due to licensing restrictions, we can only provide the `CoNLL03` dataset along with related prompts. Both the original and the processed datasets are placed in the `datasets` folder. The processed data is in JSONL format, and each entry contains the keys "text" and "entity_labels", as shown below:
```json
{"text": "EU rejects German call to boycott British lamb .", "entity_labels": [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]}
```

## Running
The commands below are executed from the repository root and use `Llama-3.1-8B-Instruct` plus `conll2003` as an example.

### 1. Use the LLM to generate rules from the training set

This step reads `datasets/conll2003/train.jsonl`, generates candidate rules, validates them on the training split, and writes the final summary rules to `datasets/conll2003/Llama-3.1-8B-Instruct_summaryrules.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python -u rule_summary.py \
  --dataset_name conll2003 \
  --model_name Llama-3.1-8B-Instruct \
  --temperature 0 \
  --top_p 1
```

Main outputs:

- `datasets/conll2003/Llama-3.1-8B-Instruct_rules.txt`
- `datasets/conll2003/Llama-3.1-8B-Instruct_validrules.txt`
- `datasets/conll2003/Llama-3.1-8B-Instruct_summaryrules.json`

### 2. Fine-tune `deberta-v3-base` on the dataset

This step trains on `train`, evaluates on `dev`, and saves the best checkpoint to `../model/deberta-v3-base/deberta_ner_conll2003/checkpoint-best`.

```bash
CUDA_VISIBLE_DEVICES=0 python tagging/scripts/train_ner.py \
  --config tagging/configs/deberta_ner_conll2003.json
```

Main outputs:

- `../model/deberta-v3-base/deberta_ner_conll2003/checkpoint-best`
- `../model/deberta-v3-base/deberta_ner_conll2003/artifacts`
- `../model/deberta-v3-base/deberta_ner_conll2003/eval/eval_metrics.json`

### 3. Build the guideline prototypes with the fine-tuned DeBERTa checkpoint

This step encodes the support examples in `Llama-3.1-8B-Instruct_summaryrules.json` and pools them into rule-level prototypes for retrieval.

```bash
CUDA_VISIBLE_DEVICES=0 python tagging/scripts/build_guideline_prototypes.py \
  --config tagging/configs/deberta_ner_conll2003.json \
  --checkpoint-path ../model/deberta-v3-base/deberta_ner_conll2003/checkpoint-best \
  --guideline-path datasets/conll2003/Llama-3.1-8B-Instruct_summaryrules.json
```

Main outputs:

- `../model/deberta-v3-base/deberta_ner_conll2003/guideline_retrieval/prototypes/vectors.npy`
- `../model/deberta-v3-base/deberta_ner_conll2003/guideline_retrieval/prototypes/metadata.jsonl`
- `../model/deberta-v3-base/deberta_ner_conll2003/guideline_retrieval/prototypes/summary.json`

### 4. Run test-time inference on the test split

This step reads only the raw test text, retrieves the most relevant guideline prototypes with the fine-tuned DeBERTa model, and then uses `Llama-3.1-8B-Instruct` to produce final NER predictions.

```bash
CUDA_VISIBLE_DEVICES=0 python run_withrule.py \
  --dataset_name conll2003 \
  --model_name Llama-3.1-8B-Instruct \
  --tagging_config tagging/configs/deberta_ner_conll2003.json \
  --ner_checkpoint_path ../model/deberta-v3-base/deberta_ner_conll2003/checkpoint-best \
  --prototype_dir ../model/deberta-v3-base/deberta_ner_conll2003/guideline_retrieval/prototypes
```

Main output:

- `datasets/conll2003/Llama-3.1-8B-Instruct_withrule_retrieval_result_detail.jsonl`

### 5. Compute the final test results

This step is the only stage that reads test labels and computes the final Precision / Recall / F1.

```bash
python ner_evaluate.py \
  --dataset_name conll2003 \
  --model_name Llama-3.1-8B-Instruct \
  --result_file datasets/conll2003/Llama-3.1-8B-Instruct_withrule_retrieval_result_detail.jsonl
```

### Optional wrapper scripts

The repository also provides shell wrappers for the two main sub-pipelines:

```bash
bash run.sh
bash tagging/run.sh train
bash tagging/run.sh guideline-build \
  --guideline-path datasets/conll2003/Llama-3.1-8B-Instruct_summaryrules.json
```

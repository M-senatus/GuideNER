# AGENTS.md

## Project scope
This repository is for deep learning research experiments rather than product software development.

Typical tasks include:
- LLM API experiments
- local / server inference
- model fine-tuning
- evaluation and ablation
- error analysis
- experiment logging and reproducibility

Priority:
1. correctness
2. reproducibility
3. cost control
4. minimal, auditable changes

---

## Core working rules
- Prefer small, explicit, reviewable changes.
- Do not silently change experiment assumptions.
- Analyze first, then propose a short plan, then implement.
- For non-trivial tasks, summarize:
  - goal
  - affected files
  - assumptions
  - validation method
- Do not refactor unrelated code.
- Do not rename files, directories, functions, or configs unless clearly necessary.

---

## Cost and safety constraints
- Never start full-scale training by default.
- Never launch large real API batches by default.
- Never delete checkpoints, logs, outputs, or datasets unless explicitly asked.
- Never overwrite raw data.
- Never modify secrets, tokens, `.env`, SSH settings, or account credentials unless explicitly asked.
- If a task may consume significant GPU time, API budget, or storage, first provide:
  1. the intended command
  2. expected scale
  3. expected cost/risk
  4. a small-scale validation path

---

## Repository assumptions
Unless the repository already defines a different convention, treat directories like this:
- `src/`: core Python code
- `configs/`: training / inference / evaluation configs
- `scripts/`: shell or orchestration scripts
- `data/`: raw or referenced data; do not modify raw source files
- `processed_data/`: generated or transformed datasets
- `eval/`: metrics and evaluation logic
- `logs/`: runtime logs
- `outputs/`: experiment outputs
- `checkpoints/`: saved model checkpoints
- `docs/`: notes and experiment documentation
- `notebooks/`: exploratory analysis only, not the main production path

If the actual repository structure differs, infer the real structure from existing files and follow the repository's own pattern.

---

## Task workflow
For any task more complex than a tiny edit, follow this order:
1. inspect relevant files
2. explain current understanding briefly
3. propose a short implementation plan
4. make minimal changes
5. run the smallest meaningful validation
6. report:
   - modified files
   - commands run
   - results
   - remaining risks or open questions

If the task changes experiment state, update project documentation if such files exist.

---

## Rules for LLM API experiments
- Keep provider, model, base_url, and generation parameters configurable.
- Add timeout, retry, and logging where appropriate.
- Prefer adding cache or resumable output logic for expensive calls.
- Prefer a `max_samples` or similar guard for batch experiments.
- Real API calls should be easy to disable.
- Store outputs in files, not only stdout.
- If adding prompt templates, keep them separate from core execution logic.

---

## Rules for inference
- Always support a tiny-run or small-sample mode.
- Keep input path, output path, batch size, device, dtype, and max_samples configurable.
- Save outputs under a structured directory, preferably with date and experiment name.
- Separate normal inference from benchmark code.
- When relevant, log:
  - model name
  - checkpoint
  - batch size
  - precision
  - device
  - runtime / throughput

---

## Rules for fine-tuning
- Before training, check dataset schema and config consistency.
- Distinguish clearly between:
  - dry-run
  - smoke test
  - tiny-subset training
  - full training
- Prefer validating with a tiny subset before proposing a full run.
- Save or record the exact config used for each run.
- Do not assume full training should start automatically after code edits.
- If training scripts are modified, provide the smallest validation command first.

---

## Rules for evaluation and ablation
- Metrics code must state its input assumptions clearly.
- Keep evaluation scripts deterministic where practical.
- Export machine-readable results when possible.
- When comparing experiments, state the exact config differences.
- For error analysis, save sampled bad cases to files for later review.
- Avoid changing metric definitions unless explicitly requested.

---

## Data handling
- Treat raw data as read-only.
- Write transformed datasets to a separate location.
- Do not silently change label schema, split logic, or preprocessing rules.
- If data format assumptions are unclear, inspect samples first and report findings before changing code.

---

## Dataset split isolation
The `train`, `dev`, and `test` splits must remain strictly isolated. These are hard constraints for any code related to data processing, training, validation, testing, caching, prompt construction, retrieval, feature engineering, preprocessing, analysis, or experiment orchestration. Do not violate them for convenience.

### Allowed usage by split
- `train` is only for model training.
- `dev` is only for validation, hyperparameter tuning, early stopping, model selection, and intermediate analysis.
- During training and validation work, it is allowed to read `train` and `dev` text and labels.

### Test split restrictions
- The `test` split is only for the final test workflow.
- In `test_infer`, only raw `test` text may be read.
- In `test_infer`, reading any `test` ground-truth field is forbidden, including entity boundaries, entity types, BIO/BIOES tags, span labels, or JSON fields such as `label` / `type`.
- In `test_infer`, the code may only generate predictions. It must not compute metrics that depend on ground truth.
- In `final_eval`, and only in `final_eval`, the code may read both `test` text and `test` labels in order to compute final metrics such as Precision / Recall / F1.
- Outside `final_eval`, no code may read `test` labels for any purpose.

### Forbidden access outside final test
In any non-final-test stage, including training, validation, tuning, error analysis, sample inspection, data statistics, prompt construction, retrieval index building, rule generation, feature engineering, cache preprocessing, or any intermediate pipeline step:
- do not read `test` text
- do not read `test` labels
- do not load any `test` content into memory, caches, indices, vector stores, or intermediate files
- do not use `test` to build vocabularies, label sets, entity type descriptions, templates, examples, or prior knowledge

### Required code-level enforcement
These constraints must be enforced in code, not only in comments or documentation.

#### Required data-loading interfaces
At minimum, split data-loading entry points as follows:
- `load_train_data()`
- `load_dev_data()`
- `load_test_text_only()`
- `load_test_with_labels_for_final_eval()`

Hard requirements:
- `load_test_text_only()` must return text-only test data and must never expose label fields.
- `load_test_with_labels_for_final_eval()` may only be called in `final_eval` mode.

#### Required stage control
Programs that may touch split data must use an explicit stage variable, for example:
- `stage = "train"`
- `stage = "dev"`
- `stage = "test_infer"`
- `stage = "final_eval"`

Enforce hard checks:
- only `stage == "test_infer"` may read test text without labels
- only `stage == "final_eval"` may read test labels
- any other stage that attempts to access `test` must raise an exception immediately

#### No implicit leakage
Avoid these invalid implementations:
- reading `train` / `dev` / `test` all at startup
- a `Dataset` or `DataLoader` that loads all splits by default
- preprocessing `test` in advance through a shared unified preprocessing script
- writing `test` labels into cache files or artifacts such as `pickle`, `npy`, `pt`, `json`, `faiss`, or vector stores
- leaking `test` labels through logs, debug output, or sample printing

#### Fail closed
If the code detects any of the following, it must raise an exception and stop immediately. Do not fall back, warn-and-continue, or silently downgrade behavior:
- a non-test stage attempts to read `test`
- `test_infer` attempts to read `test` labels
- any stage other than `final_eval` calls the full test loader

### Code writing requirements
When implementing or modifying related code:
- document the permission boundary of every data-loading function
- make test-related function names and comments explicitly distinguish text-only inference from label-based final evaluation
- do not add test-label reading logic for debugging unless the user explicitly requests it
- if an implementation could risk leaking test information, choose the more conservative design

### Required contents in code outputs
When providing code for data processing, training, validation, or testing, do not omit the protective logic. The code must clearly show:
1. separate entry points for reading `train`, `dev`, and `test`
2. separate flows for `test_infer` and `final_eval`
3. explicit assertions or exceptions that prevent accidental test-label access
4. hard enforcement of these protections rather than comment-only guidance

---

## Validation policy
Unless explicitly told otherwise, validate with the smallest meaningful check:
- syntax / import check
- one smoke test
- one tiny inference run
- one tiny training step or tiny subset run
- one evaluation smoke test

Do not run expensive end-to-end workloads by default.

If validation cannot be run, explain why clearly.

---

## Coding style
- Follow the repository's existing style first.
- Prefer clarity over cleverness.
- Add comments only where they materially improve understanding.
- Avoid unnecessary abstraction.
- Keep function and script interfaces explicit.
- Do not introduce new dependencies unless justified.-
- When adding new code, also add helpful comments where they improve readability.
- When modifying existing code, update or add comments if the logic becomes more complex.
- Prefer comments that explain:
  - the purpose of a function, class, or module
  - the meaning of important inputs / outputs
  - non-obvious logic
  - assumptions, constraints, and edge cases
  - why a particular implementation choice was made
- Do not add redundant comments that merely restate obvious code line by line.
- For simple and self-explanatory code, keep comments minimal.
- If a function is important, prefer adding a short docstring.
- If a script is used for training, inference, evaluation, or data processing, add a brief header comment explaining its role.
- When code behavior changes, make sure related comments stay consistent with the implementation.

---

## Communication style
When reporting results:
- be concise
- be concrete
- list changed files
- state what was validated
- state what remains uncertain

Do not claim success without evidence from code inspection or validation.

---

## Definition of done
A task is complete only if:
- the requested change is implemented
- modified files are identified
- the smallest relevant validation was run, or the reason it was not run is stated
- major risks / assumptions are stated
- experiment-impacting changes are documented when appropriate

# llm_metacognition

# Metacognitive Reasoning Evaluation Pipeline

A multi-stage pipeline for evaluating the metacognitive reasoning quality of large language models on programming problems.

## Data Format

### `data/code_problems.csv`
Required columns:

| Column       | Description                        |
|--------------|------------------------------------|
| `problem_id` | Unique identifier, e.g. `P01`      |
| `problem`    | Full problem statement (text)      |
| `domain`     | Problem domain, e.g. `algorithms`  |

### `data/tests/<problem_id>/`
One sub-folder per problem containing paired `inputN.txt` / `outputN.txt` files (N = 1, 2, …).

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

`.env` contents:
```
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=...
AZURE_OPENAI_DEPLOYMENT=...
```

### 3. Local models (Ollama)

Install [Ollama](https://ollama.com) and pull the required models:

```bash
ollama pull deepseek-r1:8b
ollama pull qwen2.5-coder:7b
ollama pull cogito:8b
ollama pull llama3.1:8b
ollama pull ibm/granite3.3:8b
```

---

## Running the Pipeline

All scripts are run from the **project root**. Each stage is independent — run in the order shown.

### Stage 1 — Solution Generation

**Local model (Ollama):**
```bash
python src/solution_generation/local_solver.py \
    --input_csv  data/code_problems.csv \
    --output_dir output/solver_local
```

**Commercial model:**
```bash
python src/solution_generation/commercial_solver.py \
    --model      gemini-2.5-flash \
    --input_csv  data/code_problems.csv \
    --output_dir output
# Available models: gemini-2.5-flash, mistral-large
```

Outputs per run:
- `output/.../traces.csv` — thinking traces + extracted code
- `output/.../solutions/*.py` — extracted code files

---

### Stage 2 — Code Correctness

```bash
python src/code_correctness/code_accuracy.py \
    --sol_dir    output/solver_local/solutions \
    --test_dir   data/tests \
    --output_csv output/code_correctness/results.csv
```

Output: `results.csv` — pass/fail per (solution × test case).

---

### Stage 3 — CoT Faithfulness Check

```bash
python src/faithfulness_check/cfc.py \
    --traces_csv   output/solver_local/traces.csv \
    --problems_csv data/code_problems.csv \
    --output_dir   output/cfc
```

Output: `cfs_scores.csv` — `code_aligned: True/False` per trace.

---

### Stage 4 — Metacognitive Rubric Judging

**Multi-model local panel (3 judges, Krippendorff IRR):**
```bash
python src/judge/jury.py
# Edit JUDGE_MODELS and entry-point paths at the bottom of the file.
```

**Azure OpenAI judge (3× self-consistency):**
```bash
python src/judge/judge.py \
    --traces_csv   output/solver_local/traces.csv \
    --problems_csv data/code_problems.csv \
    --output_dir   output/judge_azure
```

Outputs:
- `judge_scores.csv` — per-call scores (Q1–Q16) with rationale
- `final_scores.csv` — mean scores across calls (input for MRI stage)

> **Note:** Scores are `NaN` where a judge did not return a parseable score for a question.

---

### Stage 5 — MRI Aggregation

```bash
python src/mri_calculations.py \
    --judge_dir  output/judge_azure \
    --output_csv output/mri/mri_summary.csv
```

The script recursively finds all `final_scores.csv` files under `--judge_dir`, so you can point it at a root directory containing multiple model sub-folders.

Outputs:
- `mri_summary.csv` — `MRI_mean`, `MRI_std`, `MRI_ci`, `n_obs` per model
- `dimension_summary.csv` — mean `S_P` (Planning), `S_M` (Monitoring), `S_E` (Evaluation) per model

---

## Rubric Overview

The 16-item rubric is defined in `src/prompts.py` and covers three metacognitive dimensions:

| Dimension          | Items    | Description                              |
|--------------------|----------|------------------------------------------|
| Planning (`S_P`)   | Q1 – Q6  | Problem analysis, decomposition, sketching |
| Monitoring (`S_M`) | Q7 – Q11 | Vigilance, intermediate verification     |
| Evaluation (`S_E`) | Q12 – Q16 | Constraint checking, correctness, reflection |

Each item is scored 1–3: **1** = Not Observed, **2** = Partial, **3** = Clear.

MRI = (S_P + S_M + S_E) / 3

---

## Output Directory Layout

```
output/
├── solver_local/
│   ├── traces.csv
│   └── solutions/
├── code_correctness/
│   └── results.csv
├── cfc/
│   └── cfs_scores.csv
├── judge_azure/
│   ├── judge_scores.csv
│   └── final_scores.csv
└── mri/
    ├── mri_summary.csv
    └── dimension_summary.csv
```

---

## Notes

- All scripts support **checkpoint/resume** — re-running picks up where it left off.
- Missing judge scores are stored as `NaN` (not imputed).
- `logprob_utils.py` and `schemas.py` are required in `src/` but are not included in this distribution.
# Reproducing Paper Results

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for full training; CPU works for `--quick`)
- OpenAI-compatible API access (only needed to re-run validation; not needed for classification)

## Quick Start (< 5 minutes, CPU)

```bash
git clone <repo-url>
cd VIOLA
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/reproduce_paper_results.py --quick --skip-preprocess
```

The `--quick` flag uses 5 epochs and a small model. Results won't match the paper but
verify the full pipeline runs end-to-end.

## Full Reproduction (~2 hours on a single GPU)

```bash
python scripts/reproduce_paper_results.py --skip-preprocess
```

This trains TraceGNN and all four graph baselines on the train split and
evaluates on the test split. Outputs a results table and saves
`results/paper_results.json`.

## Step-by-Step

### 1. Download the dataset

```bash
python scripts/download_dataset.py
```

Downloads from HuggingFace to `data/`. Includes parquet splits, raw OTel logs,
policy files, and metadata.

### 2. (Optional) Re-run preprocessing

The raw OTel logs are included in the dataset download. Preprocessing converts
them to PyG graph objects for GNN training.

```bash
python scripts/train_gnn.py --preprocess --data-dir data/processed
```

This takes ~20–60 minutes depending on hardware. Pre-computed graphs are already
included in the dataset download, so you can skip this step.

### 3. Train TraceGNN

```bash
python scripts/train_gnn.py --data-dir data/processed --output-dir results/gnn
```

Checkpoints are saved to `results/gnn/`. Training logs are in `results/gnn/training.log`.

### 4. (Optional) Re-run validation pipeline

To validate newly generated traces (not needed for paper reproduction):

```bash
export JUDGE_MODEL="gpt-4.1"
export JUDGE_BASE_URL="https://api.openai.com/v1"
export JUDGE_API_KEY="sk-..."

python scripts/run_validation_pipeline.py \
    --run-dir /path/to/your/run_dir \
    --output-dir results/validation
```

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `JUDGE_MODEL` | LLM model for validation judges | `gpt-4.1` (auto-probed) |
| `JUDGE_BASE_URL` | OpenAI-compatible API base URL | OpenAI default |
| `JUDGE_API_KEY` | API key | `OPENAI_API_KEY` fallback |
| `POLICIES_DIR` | Path to original system prompt `.md` files | `data/policies/original/` |

## Expected Results

See Table 2 in the paper. The main metric is F1 averaged across agent-identification
and violation-type classification heads. TraceGNN substantially outperforms all
baselines on violation-type classification.

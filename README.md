# VIOLA: Companion Code

Companion code for *"VIOLA: A Curated Dataset for Evaluating Agent Policy Conformance"* (NeurIPS 2026 Evaluations & Datasets Track).

**Dataset** · [HuggingFace](https://huggingface.co/datasets/policy-violation-benchmark/VIOLA) · [Croissant metadata](dataset/croissant.json)

---

## Quickstart

```bash
git clone <repo-url>
cd VIOLA
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/download_pretrained.py
python eval.py --model checkpoints/gnn_main.pt --split test
```

For a CPU-friendly sanity check (~5 min):
```bash
python train.py --quick
```

---

## Requirements

**Python**: 3.10+

```bash
pip install -r requirements.txt
```

**GPU**: CUDA-capable GPU recommended for full training. The `--quick` flag is CPU-feasible (~5 min).

**For the validation pipeline only** (not needed for classification experiments):
```bash
export JUDGE_MODEL="gpt-4.1"
export JUDGE_BASE_URL="https://api.openai.com/v1"
export JUDGE_API_KEY="sk-..."
```

---

## Dataset

```bash
python scripts/download_dataset.py
```

Downloads all splits (train/val/test), raw OTel logs, policy files, and metadata from HuggingFace to `data/`.

- Dataset card: [huggingface.co/datasets/policy-violation-benchmark/VIOLA](https://huggingface.co/datasets/policy-violation-benchmark/VIOLA)
- Croissant metadata: [`dataset/croissant.json`](dataset/croissant.json)
- 400 traces across 11 violation types and 5 agents; 63 clean control traces

See [docs/DATASET.md](docs/DATASET.md) for the full schema and field descriptions.

---

## Training

Train TraceGNN from scratch:

```bash
python train.py                          # full run (~2 hr, single GPU)
python train.py --quick                  # 5 epochs, small model, ~5 min CPU
```

Train baselines:

```bash
python scripts/train_baselines.py --baseline all
python scripts/train_baselines.py --baseline gcn --quick
```

**Expected runtime**: ~2 hours on a single NVIDIA A100 for the full GNN run.
Checkpoints are saved to `results/gnn/best_model.pt`.

---

## Evaluation

```bash
# Evaluate released GNN checkpoint
python eval.py --model checkpoints/gnn_main.pt --split test

# Evaluate a locally trained checkpoint
python eval.py --model results/gnn/best_model.pt --split test

# Evaluate a baseline
python eval.py --model checkpoints/gcn_baseline.pt --split test
```

---

## Pre-trained Models

Download all released checkpoints:

```bash
python scripts/download_pretrained.py
```

| Checkpoint | Description | Command |
|---|---|---|
| `checkpoints/gnn_main.pt` | TraceGNN — best single run (seed 45) | `python eval.py --model checkpoints/gnn_main.pt` |
| `checkpoints/gcn_baseline.pt` | GCN baseline (seed 42) | `python eval.py --model checkpoints/gcn_baseline.pt` |
| `checkpoints/sage_baseline.pt` | GraphSAGE baseline (seed 42) | `python eval.py --model checkpoints/sage_baseline.pt` |
| `checkpoints/mlp_baseline.pt` | MLP (mean-pool) baseline (seed 42) | `python eval.py --model checkpoints/mlp_baseline.pt` |
| `checkpoints/linear_baseline.pt` | Linear (mean-pool) baseline (seed 42) | `python eval.py --model checkpoints/linear_baseline.pt` |

---

## Results

Results on the **test split** (400 traces, stratified by violation × agent):

Results are mean ± std over 5 random seeds (42–46). The released checkpoint for each model is the best single seed.

| Model | Agent Acc (%) | Agent F1 (%) | Violation Acc (%) | Violation F1 (%) | Command |
|---|:---:|:---:|:---:|:---:|---|
| **TraceGNN** (ours) | **49.0 ± 2.1** | 32.6 ± 2.3 | **26.6 ± 4.0** | **11.0 ± 2.1** | `python eval.py --model checkpoints/gnn_main.pt` |
| GCN | 48.6 ± 4.0 | 31.4 ± 3.4 | 25.9 ± 5.2 | 9.1 ± 3.1 | `python eval.py --model checkpoints/gcn_baseline.pt` |
| GraphSAGE | 46.9 ± 2.3 | 29.4 ± 3.6 | 25.5 ± 4.7 | 8.8 ± 2.4 | `python eval.py --model checkpoints/sage_baseline.pt` |
| MLP (mean-pool) | 41.4 ± 0.0 | 9.8 ± 0.0 | 17.2 ± 0.0 | 2.5 ± 0.0 | `python eval.py --model checkpoints/mlp_baseline.pt` |
| Linear | 38.6 ± 5.8 | **34.5 ± 5.1** | 15.5 ± 3.1 | 7.5 ± 1.7 | `python eval.py --model checkpoints/linear_baseline.pt` |
| GPT-4.1 (zero-shot) | 6.9 | 3.3 | 5.2 | 2.4 | `python scripts/run_llm_baseline.py` |

The released `gnn_main.pt` checkpoint (seed 45) achieves: Agent Acc 50.0%, Agent F1 35.7%, Violation Acc 25.9%, Violation F1 15.1%.

---

## Reproducing Paper Results

See [docs/REPRODUCING.md](docs/REPRODUCING.md) for full step-by-step instructions.

Fast path using pre-trained checkpoints (<30 min):
```bash
python scripts/download_pretrained.py
python scripts/reproduce_paper_results.py
```

Full from-scratch path (~2 hr, single GPU):
```bash
python scripts/reproduce_paper_results.py --from-scratch
```

---

## Repository Structure

```
VIOLA/
├── train.py                        Top-level training entry point (PwC item 2)
├── eval.py                         Top-level evaluation entry point (PwC item 3)
├── requirements.txt
├── setup.py
├── dataset/
│   └── croissant.json              Croissant ML metadata (NeurIPS requirement)
├── docs/
│   ├── DATASET.md                  Dataset schema and field descriptions
│   ├── REPRODUCING.md              Step-by-step reproduction instructions
│   └── ARCHITECTURE.md             Code architecture overview
├── src/
│   ├── generation/                 Violation distorters (V1–V11)
│   ├── validation/                 4-stage LLM-judge validation pipeline
│   ├── extraction/                 OTel span extraction
│   ├── classification/             TraceGNN: preprocessing + training
│   └── baselines/                  Graph baselines + LLM zero-shot baseline
├── scripts/
│   ├── download_dataset.py         Download dataset from HuggingFace
│   ├── download_pretrained.py      Download pre-trained checkpoints
│   ├── train_baselines.py          Train graph baselines
│   ├── run_validation_pipeline.py  Validate custom OTel traces
│   └── reproduce_paper_results.py  End-to-end paper reproduction
├── configs/
│   ├── default.yaml                Default training hyperparameters
│   └── ablations/small.yaml        Fast config for sanity checks
├── checkpoints/                    Downloaded checkpoints go here (gitignored)
├── tests/
│   ├── test_extraction.py
│   ├── test_validation.py
│   └── test_classification.py
└── results/                        Training outputs (gitignored)
```

---

## Limitations & External Dependencies

The following dependencies depart from full self-containment. Per the NeurIPS CFP, these are documented here rather than silently worked around:

**LLM API calls (validation pipeline only)**: The 4-stage validation pipeline (Judge 1 / Judge 2) requires an OpenAI-compatible API. This API is needed only to *reproduce dataset construction* — not to reproduce the classification experiments, which are the scientific contribution of the paper. The dataset is already published on HuggingFace, so all classification results can be reproduced without any API calls.

**GPU for full training**: TraceGNN training is feasible on a single consumer GPU (~2 hr on A100). The `--quick` flag (`python train.py --quick`) runs in ~5 minutes on CPU for sanity checking.

**AppWorld environment**: Generating new violation traces requires the AppWorld evaluation harness. This is only needed to *extend* the dataset; reproducing the paper results does not require it.

---

## Citation

```bibtex
@misc{viola2026,
  title   = {VIOLA: A Curated Dataset for Evaluating Agent Policy Conformance},
  author  = {Anonymous},
  year    = {2026},
  note    = {NeurIPS 2026 Evaluations \& Datasets Track.
             Dataset: https://huggingface.co/datasets/policy-violation-benchmark/VIOLA}
}
```

---

## License

Code: [MIT](LICENSE) · Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

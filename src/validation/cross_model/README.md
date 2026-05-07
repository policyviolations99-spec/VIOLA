# Cross-model judge validation

Re-runs the existing J1 (theoretic) and J2 (executional) judges on a
stratified subset of the published benchmark using two additional model
families — Claude Sonnet 4.6 and Gemini 2.5 Pro — and reports inter-judge
agreement metrics for the paper appendix.

## How it routes

All three judges go through a **single OpenAI-compatible endpoint** (e.g. a
litellm proxy or any gateway that fronts Azure / Anthropic / Google with the
same API key). Only the `model` field changes between runners:

| Judge | Default model id |
|---|---|
| GPT (baseline) | `Azure/gpt-4.1` |
| Claude         | `claude-sonnet-4-6` |
| Gemini         | `GCP/gemini-2.5-pro` |

If your gateway exposes the models under different names, override via
`CROSS_MODEL_GPT_ID`, `CROSS_MODEL_CLAUDE_ID`, `CROSS_MODEL_GEMINI_ID`.

## Required environment

```bash
# Endpoint + key (same as the existing GPT-4.1 pipeline)
export JUDGE_BASE_URL="https://your-litellm-proxy"     # or OPENAI_BASE_URL
export JUDGE_API_KEY="sk-..."                          # or OPENAI_API_KEY

# Where to find the dataset (manifest.json + logs/)
export BENCHMARK_RUN_DIR="/path/to/run_dir"

# Where the existing GPT-4.1 _validation.json files live
export GPT_VALIDATION_DIR="/path/to/results/validation_output"

# Optional: redirect outputs (default: alongside this package)
export CROSS_MODEL_OUTPUT_DIR="/path/to/output_dir"
```

## Required Python packages

```bash
pip install openai pandas scikit-learn statsmodels
```

`statsmodels` is only needed for three-way Fleiss' kappa.

## How to run

```bash
# Build the stratified sample only (no API calls)
python -m src.validation.cross_model.run_cross_model_validation --sample-only

# Full run (sample + Claude + Gemini + metrics + paper tables)
python -m src.validation.cross_model.run_cross_model_validation

# Just the metrics step (re-aggregate an existing JSONL)
python -m src.validation.cross_model.run_cross_model_validation --metrics-only

# Skip a provider
python -m src.validation.cross_model.run_cross_model_validation --skip-gemini
```

The orchestrator caches every successful (trace, model, stage, run_idx)
record into `results/cross_model_judgments.jsonl`. Re-running the script
picks up where the previous run left off.

## Notes on Gemini 2.5 Pro

Gemini 2.5 Pro is a reasoning model and most gateways reject
`thinking_budget=0`. We therefore send `reasoning_effort: "minimal"` via
`extra_body` and allocate `max_tokens=4096` for the Gemini runner so the
visible JSON output isn't starved by hidden reasoning tokens. (Claude and
GPT use the standard 1024-token budget mirroring the existing pipeline.)

## Outputs

```
results/
  sample_manifest.json            # sampled traces (deterministic from seed=42)
  cross_model_judgments.jsonl     # raw per-call records (append-only cache)
  aggregated_labels.csv           # one row per (trace, judge): consensus + acceptance
paper_tables/
  pairwise_kappa.tex              # Cohen's kappa per judge pair, with 95% CI
  fleiss_kappa.tex                # three-way Fleiss' kappa per stage
  per_type_kappa.tex              # J2 kappa broken out by violation type
  acceptance_overlap.tex          # acceptance-decision confusion per judge pair
  reclassification_agreement.tex  # hard/soft label agreement on doubly-accepted traces
  disagreement_examples.md        # 10 most contentious traces (appendix material)
```

## What gets sampled

Default config (see `config.py`):

* 110 GPT-accepted traces — 10 per active violation type, balanced 50/50
  hard/soft where possible.
* 30 GPT-rejected traces (precheck-passed) stratified by rejection reason.
* 10 clean controls — for each clean run we deterministically pick one
  `(violation_id, agent)` target so a successful judge response should be
  `policy_present` / `no_violation` for every model (negative control).

Sample size lands around 140-150 traces depending on availability per
violation type. Reproducible from `SAMPLING_SEED = 42`.

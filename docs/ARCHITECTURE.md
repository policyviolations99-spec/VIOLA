# Code Architecture

## Overview

```
src/
├── generation/      Violation distorters: inject contrary instructions into agent prompts
├── validation/      4-stage validation pipeline: precheck + 2 LLM judges + consensus
├── extraction/      OTel span extraction: parse raw logs into (prompt, input, response)
├── classification/  GNN classifier: preprocess traces into graphs, train TraceGNN
└── baselines/       Comparison baselines: Linear, MLP, GCN, GraphSAGE, LLM-zero-shot
```

## Generation (`src/generation/`)

Each of the 11 violation types has its own distorter class in `distorters/`.
All inherit from `distorters.base.ViolationDistorter`, which dispatches to
per-agent `_apply_{AgentName}()` methods.

The contrary instruction pattern: each distorter replaces a specific policy clause
in the agent's system prompt with an actively misdirecting contrary instruction
(rather than simply deleting it). The `COMPATIBILITY_MATRIX` encodes which
(violation, agent) pairs have a policy to target.

## Validation (`src/validation/`)

**Stage 1 — Programmatic precheck** (`precheck.py`): Deterministic diff check that
the injection actually modified the prompt. Rejects if original == modified or if
no lines were removed.

**Stage 2 — Judge 1 × 3** (`judge_runner.call_judge_1`): LLM auditor checks whether
the targeted policy is semantically absent from the modified prompt. Verdict:
`policy_absent` / `policy_present`. Pass threshold: ≥2/3 calls return `policy_absent`.

**Stage 3 — Judge 2 × 3** (`judge_runner.call_judge_2`): LLM evaluator checks whether
the agent response exhibits the expected violation behavior. Uses per-cell behavioral
indicators from `violation_config.py`. Verdict: `hard_violation` / `soft_violation` /
`no_violation`. Pass threshold: ≥2/3 calls return a violation label.

**Stage 4 — Consensus** (`consensus.py`): Aggregates all results. Both judges must
pass. Final label is the majority Judge 2 verdict.

## Classification (`src/classification/`)

**Preprocessing** (`preprocessing/`): Converts raw OTel span logs into PyG `Data`
objects. Each node is either an LLM call span (with prompt/response embeddings) or
a non-LLM span (with structural features). Edges encode the span execution order.

Key components:
- `node_extraction.py`: parses spans from log files
- `llm_encoding/`: embeds LLM call content using sentence-transformers
- `non_llm_encoding/`: computes structural/metadata features for non-LLM spans
- `output/pyg_converter.py`: assembles node features + edges into PyG Data objects

**TraceGNN** (`training/model.py`): Heterogeneous GAT architecture:
1. `NodeTypeEncoder`: separate linear encoders for LLM vs. non-LLM nodes → common hidden dim
2. `GATLayer × N`: graph attention propagation
3. Global pooling (mean + max + sum concat)
4. Two classification heads: agent-identification (5-class) + violation-type (12-class)

## Baselines (`src/baselines/`)

`graph_baselines.py` provides four baselines that share the same dual-head interface:
- `LinearBaseline`: mean-pool + logistic regression (no hidden layers)
- `MLPBaseline`: mean-pool + 2-layer MLP (no graph structure)
- `GCNBaseline`: standard GCN (Kipf & Welling 2017)
- `GraphSAGEBaseline`: inductive neighborhood aggregation

`llm_baseline.py`: zero-shot LLM classifier that reads the modified system prompt and
predicts the violation type without any training.

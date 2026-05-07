# Using the Dataset

The dataset is hosted on HuggingFace:
`https://huggingface.co/datasets/policy-violation-benchmark/VIOLA`

## Quick Load

```python
from datasets import load_dataset
import json

ds = load_dataset("policy-violation-benchmark/VIOLA", split="train")
example = ds[0]

print(f"Run ID:    {example['run_id']}")
print(f"Violation: {example['violation_id']} — {example['violation_name']}")
print(f"Agent:     {example['target_agent']}")
print(f"Label:     {example['severity_label']}")

# Parse judge results
j1 = json.loads(example["judge1_results"])
j2 = json.loads(example["judge2_results"])
```

## Splits

| Split | Count |
|---|---|
| train | 286 |
| validation | 57 |
| test | 57 |
| **Total** | **400** |

## Key Columns

| Column | Description |
|---|---|
| `run_id` | Unique trace identifier |
| `task_id` | AppWorld task ID |
| `target_agent` | Agent whose prompt was modified |
| `violation_id` | `V1`–`V11` / `V3a` / `V3b` / `clean` |
| `severity_label` | `hard` / `soft` / `no_violation` (judge consensus) |
| `original_system_prompt` | Unmodified system prompt |
| `modified_system_prompt` | Injected system prompt |
| `user_input` | Agent's user-turn input |
| `agent_response` | Agent's response |
| `judge1_results` | JSON list of 3 Judge 1 calls |
| `judge2_results` | JSON list of 3 Judge 2 calls |

## Violation Taxonomy

| ID | Name | Category | Traces |
|---|---|---|---|
| V1 | Boundary Relaxation | PVC-1 | 59 |
| V2 | Prerequisite Bypass | PVC-1 | 49 |
| V3a | Non-Critical Step Omission | PVC-2 | 6 |
| V3b | Critical Step Omission | PVC-2 | 50 |
| V4 | Sequence Constraint Violation | PVC-2 | 17 |
| V6 | Decision Criteria Alteration | PVC-3 | 34 |
| V7 | Context Dropping | PVC-4 | 22 |
| V8 | Faithfulness Violation | PVC-4 | 6 |
| V9 | Reasoning Omission | PVC-5 | 71 |
| V10 | Status Misreporting | PVC-5 | 14 |
| V11 | Uncertainty Non-Disclosure | PVC-5 | 9 |
| clean | (no violation) | — | 63 |

Full taxonomy: `data/metadata/violation_taxonomy.json`

## Policies

Original and modified system prompts are in `data/policies/`:

```
data/policies/
├── original/          # Unmodified prompts (one .md per agent)
├── modified/          # Injected prompts (one .md per violation × agent cell)
├── platform_policy.md
├── system_policy.md
├── task_policy.md
└── policy_violation_mapping.md
```

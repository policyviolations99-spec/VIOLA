"""
Structural features extracted from LLM JSON output.

These are per-node, response-level features (what THIS agent node output),
not per-graph aggregates.  Adding them gives the GNN access to the same
boolean/numeric signals that the LR baseline uses, without leaking which
agent was the injection site.

Feature layout (12 dims):
  [0]  action_is_coder       — action == "CoderAgent"
  [1]  action_is_conclude    — action == "ConcludeTask"
  [2]  action_is_shortlist   — action == "ShortlisterAgent"
  [3]  action_is_other       — action present but none of the above
  [4]  num_subtasks          — len(subtasks or subtask_list), log-scaled
  [5]  has_subtasks          — subtask list non-empty
  [6]  has_relevance_scores  — any api has relevance_score
  [7]  mean_relevance_score  — mean score in [0,1] (0 if none)
  [8]  has_code              — code / python_code field non-empty
  [9]  completion_fraction   — completed / total subtask_progress entries
  [10] num_relevant_apis     — len(relevant_apis), log-scaled
  [11] num_json_keys         — total top-level JSON keys, log-scaled
"""

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

STRUCTURAL_DIM = 12


def _safe_log1p(x: float) -> float:
    return float(math.log1p(max(x, 0.0)))


def extract_structural_features(output_text: Optional[str]) -> np.ndarray:
    """Parse the LLM's JSON output and return a (12,) float32 feature vector."""
    feats = np.zeros(STRUCTURAL_DIM, dtype=np.float32)

    if not output_text:
        return feats

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(output_text)
    except (json.JSONDecodeError, ValueError):
        # Non-JSON output (e.g. code): return all-zero structural features
        return feats

    if not isinstance(parsed, dict):
        return feats

    # [0-3] action one-hot
    action = parsed.get("action", "")
    feats[0] = 1.0 if action == "CoderAgent"    else 0.0
    feats[1] = 1.0 if action == "ConcludeTask"  else 0.0
    feats[2] = 1.0 if action == "ShortlisterAgent" else 0.0
    feats[3] = 1.0 if (action and not any(feats[:3])) else 0.0

    # [4-5] subtasks
    subtasks = parsed.get("subtasks", parsed.get("subtask_list", []))
    n_sub = len(subtasks) if isinstance(subtasks, list) else 0
    feats[4] = _safe_log1p(n_sub)
    feats[5] = 1.0 if n_sub > 0 else 0.0

    # [6-7] relevance scores from shortlisted_apis
    shortlisted = parsed.get("shortlisted_apis", parsed.get("apis", []))
    scores: List[float] = []
    if isinstance(shortlisted, list):
        for api in shortlisted:
            if isinstance(api, dict):
                s = api.get("relevance_score", api.get("score"))
                if s is not None:
                    try:
                        scores.append(float(s))
                    except (ValueError, TypeError):
                        pass
    feats[6] = 1.0 if scores else 0.0
    feats[7] = float(sum(scores) / len(scores)) if scores else 0.0

    # [8] has_code
    code = parsed.get("code", parsed.get("python_code", ""))
    feats[8] = 1.0 if code else 0.0

    # [9] completion_fraction
    progress = parsed.get("subtasks_progress", [])
    if isinstance(progress, list) and len(progress) > 0:
        n_done = sum(1 for s in progress if s == "completed")
        feats[9] = float(n_done / len(progress))
    else:
        feats[9] = 0.0

    # [10] num_relevant_apis
    rel_apis = parsed.get("relevant_apis", [])
    feats[10] = _safe_log1p(len(rel_apis) if isinstance(rel_apis, list) else 0)

    # [11] num_json_keys
    feats[11] = _safe_log1p(len(parsed))

    return feats


STRUCTURAL_FEATURE_NAMES = [
    "action_is_coder",
    "action_is_conclude",
    "action_is_shortlist",
    "action_is_other",
    "num_subtasks_log1p",
    "has_subtasks",
    "has_relevance_scores",
    "mean_relevance_score",
    "has_code",
    "completion_fraction",
    "num_relevant_apis_log1p",
    "num_json_keys_log1p",
]

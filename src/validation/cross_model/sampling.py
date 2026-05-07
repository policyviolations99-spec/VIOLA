"""
Stratified subset selection for cross-model validation.

  - 110 GPT-accepted traces — 10 per active violation type, balanced hard/soft.
  - 30 GPT-rejected traces (precheck-passed) stratified by rejection reason.
  - 10 clean controls — different (violation, agent) targets per trace.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from typing import Dict, List

from src.validation.models import TraceForValidation

from .config import (
    ACTIVE_VIOLATION_TYPES,
    N_ACCEPTED_PER_VIOLATION_TYPE,
    N_CLEAN_CONTROLS,
    N_REJECTED_TRACES,
    SAMPLING_SEED,
    sample_manifest_path,
)
from .trace_loader import GPTLabel

logger = logging.getLogger(__name__)


def _balance_hard_soft(
    rng: random.Random,
    type_traces: List[TraceForValidation],
    gpt_labels: Dict[str, GPTLabel],
    target: int,
) -> List[TraceForValidation]:
    hard, soft = [], []
    for t in type_traces:
        lbl = gpt_labels.get(t.run_id)
        if lbl is None:
            continue
        if lbl.final_label == "hard":
            hard.append(t)
        elif lbl.final_label == "soft":
            soft.append(t)
    rng.shuffle(hard)
    rng.shuffle(soft)
    half = target // 2
    n_hard = min(half, len(hard))
    n_soft = min(target - n_hard, len(soft))
    if n_hard + n_soft < target:
        leftover = hard[n_hard:] + soft[n_soft:]
        rng.shuffle(leftover)
        extra = leftover[: target - n_hard - n_soft]
    else:
        extra = []
    return hard[:n_hard] + soft[:n_soft] + extra


def build_stratified_sample(
    accepted_traces: List[TraceForValidation],
    rejected_traces: List[TraceForValidation],
    clean_traces: List[TraceForValidation],
    gpt_labels: Dict[str, GPTLabel],
    seed: int = SAMPLING_SEED,
) -> List[TraceForValidation]:
    rng = random.Random(seed)
    sample: List[TraceForValidation] = []
    sample_run_ids = set()

    by_type = defaultdict(list)
    for t in accepted_traces:
        if t.violation_id in ACTIVE_VIOLATION_TYPES:
            by_type[t.violation_id].append(t)

    for vtype in ACTIVE_VIOLATION_TYPES:
        type_traces = by_type.get(vtype, [])
        if not type_traces:
            logger.warning("No accepted traces for %s — skipping", vtype)
            continue
        chosen = _balance_hard_soft(
            rng, type_traces, gpt_labels, N_ACCEPTED_PER_VIOLATION_TYPE
        )
        for t in chosen:
            if t.run_id in sample_run_ids:
                continue
            sample.append(t)
            sample_run_ids.add(t.run_id)
        logger.info("Accepted/%s: %d/%d", vtype, len(chosen), len(type_traces))

    by_reason = defaultdict(list)
    for t in rejected_traces:
        lbl = gpt_labels.get(t.run_id)
        reason = (lbl.rejection_reason if lbl else "unknown") or "unknown"
        by_reason[reason].append(t)

    if by_reason:
        per_reason = max(1, N_REJECTED_TRACES // len(by_reason))
        for reason, traces in by_reason.items():
            rng.shuffle(traces)
            for t in traces[:per_reason]:
                if t.run_id in sample_run_ids:
                    continue
                sample.append(t)
                sample_run_ids.add(t.run_id)
        remaining = N_REJECTED_TRACES - sum(min(per_reason, len(v)) for v in by_reason.values())
        if remaining > 0:
            pool = [t for traces in by_reason.values() for t in traces if t.run_id not in sample_run_ids]
            rng.shuffle(pool)
            for t in pool[:remaining]:
                sample.append(t)
                sample_run_ids.add(t.run_id)

    if clean_traces:
        rng.shuffle(clean_traces)
        for t in clean_traces[:N_CLEAN_CONTROLS]:
            if t.run_id in sample_run_ids:
                continue
            sample.append(t)
            sample_run_ids.add(t.run_id)

    logger.info("Final sample size: %d", len(sample))
    return sample


def write_manifest(
    sample: List[TraceForValidation],
    gpt_labels: Dict[str, GPTLabel],
    path=None,
) -> None:
    path = path or sample_manifest_path()
    records = []
    for t in sample:
        lbl = gpt_labels.get(t.run_id)
        if t.violation_category == "control":
            stratum = "clean_control"
        elif lbl and lbl.accepted:
            stratum = f"accepted/{t.violation_id}"
        elif lbl:
            stratum = f"rejected/{lbl.rejection_reason or 'unknown'}"
        else:
            stratum = "unknown"
        records.append({
            "run_id": t.run_id,
            "task_id": t.task_id,
            "violation_id": t.violation_id,
            "target_agent": t.target_agent,
            "designed_hard_soft": t.designed_hard_soft,
            "stratum": stratum,
            "gpt_accepted": bool(lbl.accepted) if lbl else None,
            "gpt_final_label": lbl.final_label if lbl else None,
            "gpt_rejection_reason": lbl.rejection_reason if lbl else None,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"seed": SAMPLING_SEED, "n": len(records), "traces": records},
        indent=2,
    ))
    logger.info("Wrote sample manifest: %s", path)

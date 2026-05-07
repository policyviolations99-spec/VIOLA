"""
Load benchmark traces and existing GPT-4.1 validation labels.

This is the public-repo variant: it consumes a benchmark run directory in the
shape used by ``src/validation/pipeline.py`` (manifest.json + logs/) plus a
directory of per-trace ``*_validation.json`` files produced by the existing
GPT-4.1 judge pipeline. Both locations are supplied via environment variables
so the package has no hard-coded filesystem dependencies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.validation.models import TraceForValidation
from src.validation.pipeline import load_traces_from_run

from .config import (
    ACTIVE_VIOLATION_TYPES,
    benchmark_run_dir,
    gpt_validation_dir,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Existing GPT-4.1 labels
# ---------------------------------------------------------------------------

@dataclass
class GPTLabel:
    """The fields we need from an existing per-trace GPT-4.1 validation JSON."""
    run_id: str
    precheck_passed: bool
    judge1_consensus: str
    judge2_consensus: str
    accepted: bool
    final_label: Optional[str]
    rejection_reason: Optional[str]
    judge1_results: list
    judge2_results: list


def load_gpt_labels(directory: Optional[Path] = None) -> Dict[str, GPTLabel]:
    """Walk a validation result directory and return run_id -> GPTLabel."""
    directory = directory or gpt_validation_dir()
    if not directory.exists():
        raise RuntimeError(f"GPT validation directory does not exist: {directory}")

    by_run_id: Dict[str, GPTLabel] = {}
    for path in directory.glob("**/*_validation.json"):
        run_id = path.name.removesuffix("_validation.json")
        try:
            with path.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue
        # Most-recent file wins on duplicate run_ids.
        if run_id in by_run_id:
            existing_path = directory / f"{run_id}_validation.json"
            if existing_path.exists() and existing_path.stat().st_mtime > path.stat().st_mtime:
                continue
        by_run_id[run_id] = GPTLabel(
            run_id=run_id,
            precheck_passed=bool(data.get("precheck_passed", False)),
            judge1_consensus=data.get("judge1_consensus", "n/a"),
            judge2_consensus=data.get("judge2_consensus", "n/a"),
            accepted=bool(data.get("accepted", False)),
            final_label=data.get("final_label"),
            rejection_reason=data.get("rejection_reason"),
            judge1_results=list(data.get("judge1_results") or []),
            judge2_results=list(data.get("judge2_results") or []),
        )
    logger.info("Loaded %d GPT-4.1 labels (accepted=%d, rejected=%d)",
                len(by_run_id),
                sum(1 for v in by_run_id.values() if v.accepted),
                sum(1 for v in by_run_id.values() if not v.accepted))
    return by_run_id


# ---------------------------------------------------------------------------
# Trace loaders
# ---------------------------------------------------------------------------

def load_violation_traces(run_dir: Optional[Path] = None) -> List[TraceForValidation]:
    """Load all violation traces (excluding clean controls)."""
    run_dir = run_dir or benchmark_run_dir()
    traces = load_traces_from_run(str(run_dir))
    return [t for t in traces if t.violation_id in ACTIVE_VIOLATION_TYPES]


def load_clean_traces(run_dir: Optional[Path] = None) -> List[TraceForValidation]:
    """Load clean (no-violation) baseline traces, if present.

    Some manifests label clean rows as ``violation_id == 'clean'`` and others
    omit them entirely. We accept any trace whose violation_id is not in the
    active set.
    """
    run_dir = run_dir or benchmark_run_dir()
    traces = load_traces_from_run(str(run_dir))
    return [t for t in traces if t.violation_id not in ACTIVE_VIOLATION_TYPES]


def load_clean_control_traces(seed: int = 42) -> List[TraceForValidation]:
    """Build clean-control TraceForValidation records for cross-model validation.

    For every clean run we pick one ``(violation_id, agent)`` target
    deterministically (seeded by run_id). The system prompt is the unmodified
    original — every judge should return ``policy_present`` / ``no_violation``.
    """
    import random
    from src.validation.pipeline import (  # local import: avoids hard dep at module load
        AGENT_TEMPLATE_FILES,
        load_original_system_prompt,
    )
    from src.validation.violation_config import VIOLATION_POLICY_TEXTS

    rng = random.Random(seed)
    targets = []
    for vid in ACTIVE_VIOLATION_TYPES:
        for agent in VIOLATION_POLICY_TEXTS.get(vid, {}):
            if agent in AGENT_TEMPLATE_FILES:
                targets.append((vid, agent))
    if not targets:
        return []

    clean = load_clean_traces()
    out: List[TraceForValidation] = []
    for trace in clean:
        # hash() on strings is randomised per Python invocation; use a stable digest.
        import hashlib
        digest = hashlib.md5(trace.run_id.encode()).hexdigest()
        rng.seed(int(digest, 16) & 0xFFFFFFFF)
        order = list(range(len(targets)))
        rng.shuffle(order)
        for idx in order:
            vid, agent = targets[idx]
            try:
                original = load_original_system_prompt(agent)
            except (ValueError, FileNotFoundError):
                continue
            # Keep the trace's actual content but reassign the synthetic target.
            out.append(TraceForValidation(
                run_id=f"{trace.run_id}::ctrl::{vid}::{agent}",
                task_id=trace.task_id,
                violation_id=vid,
                violation_category="control",
                violation_name="clean_control",
                designed_hard_soft="none",
                target_agent=agent,
                original_system_prompt=original,
                modified_system_prompt=trace.modified_system_prompt or original,
                user_input=trace.user_input,
                agent_response=trace.agent_response,
                task_pass_percentage=trace.task_pass_percentage,
            ))
            break
    logger.info("Loaded %d clean control traces", len(out))
    return out


def load_rejected_traces(
    gpt_labels: Dict[str, GPTLabel],
    run_dir: Optional[Path] = None,
) -> List[TraceForValidation]:
    """Return GPT-rejected traces (precheck-passed) for cross-model comparison."""
    run_dir = run_dir or benchmark_run_dir()
    traces = load_traces_from_run(str(run_dir))
    rejected = []
    for t in traces:
        lbl = gpt_labels.get(t.run_id)
        if lbl is not None and not lbl.accepted and lbl.precheck_passed:
            rejected.append(t)
    logger.info("Loaded %d rejected traces", len(rejected))
    return rejected

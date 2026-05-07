"""
Main orchestrator for the validation pipeline.

Responsibilities:
  1. Load traces from a benchmark run directory (manifest.json + log files)
  2. Run precheck -> Judge 1 x3 -> Judge 2 x3 -> consensus for each trace
  3. Save per-trace results and return aggregated ValidationResult list

Trace extraction from log files
--------------------------------
Each log file contains pretty-printed OpenTelemetry spans (one JSON object per
blank-line-separated block).  We extract:

  modified_system_prompt  - gen_ai.prompt.0.content from the first ChatOpenAI.chat
                            span whose path starts with "{agent}."
  user_input              - gen_ai.prompt.1.content from the same span
  agent_response          - output of the first RunnableLambda.task span in the
                            agent's chain (parsed JSON string inside {"outputs": "..."})

  original_system_prompt  - read from the policies/original/ directory included in
                            this repository (one .md file per agent)
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from .models import TraceForValidation, ValidationResult
from .precheck import programmatic_precheck
from .judge_runner import run_judge_1_triple, run_judge_2_triple
from .consensus import aggregate_consensus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent -> original system prompt paths
# ---------------------------------------------------------------------------
# By default, reads from policies/original/ in the dataset download directory.
# Override by setting POLICIES_DIR environment variable to a local path.

_DEFAULT_POLICIES_DIR = Path(__file__).parent.parent.parent / "data" / "policies" / "original"

AGENT_TEMPLATE_FILES: Dict[str, str] = {
    "APIPlannerAgent":        "APIPlannerAgent.md",
    "APICodePlannerAgent":    "APICodePlannerAgent.md",
    "ShortlisterAgent":       "ShortlisterAgent.md",
    "TaskDecompositionAgent": "TaskDecompositionAgent.md",
    "PlanControllerAgent":    "PlanControllerAgent.md",
    "FinalAnswerAgent":       "FinalAnswerAgent.md",
}


def _get_policies_dir() -> Path:
    env = os.environ.get("POLICIES_DIR")
    if env:
        return Path(env)
    return _DEFAULT_POLICIES_DIR


def load_original_system_prompt(agent: str) -> str:
    """
    Load the original (unmodified) system prompt for the given agent.

    Reads from the policies/original/ directory, which is included in the
    dataset download (see scripts/download_dataset.py). Override the directory
    by setting the POLICIES_DIR environment variable.
    """
    filename = AGENT_TEMPLATE_FILES.get(agent)
    if filename is None:
        raise ValueError(f"No system prompt file configured for agent: {agent}")
    policy_file = _get_policies_dir() / filename
    if not policy_file.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {policy_file}\n"
            f"Run 'python scripts/download_dataset.py' to download the dataset first, "
            f"or set POLICIES_DIR to the path containing the original prompt .md files."
        )
    return policy_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Log file parsing
# ---------------------------------------------------------------------------

def _parse_spans(log_path: Path) -> List[Dict[str, Any]]:
    """Parse a log file into a list of OpenTelemetry span dicts."""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"\n(?=\{)", content)
    spans = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            obj = json.loads(part)
            if isinstance(obj, dict):
                spans.append(obj)
        except json.JSONDecodeError:
            pass
    return spans


def _span_belongs_to_agent(span: Dict[str, Any], agent: str) -> bool:
    attrs = span.get("attributes", {})
    path = attrs.get("traceloop.entity.path", "")
    cp_ns = attrs.get("traceloop.association.properties.langgraph_checkpoint_ns", "")
    node = attrs.get("traceloop.association.properties.langgraph_node", "")
    return (
        path.startswith(f"{agent}.") or
        path == agent or
        cp_ns.startswith(f"{agent}:") or
        node == agent
    )


def _parse_agent_response(raw_output: str) -> Optional[str]:
    try:
        output_wrapper = json.loads(raw_output)
        outputs = output_wrapper.get("outputs")
        if outputs is None:
            return None
        if isinstance(outputs, str):
            try:
                parsed = json.loads(outputs)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                return outputs
        elif isinstance(outputs, dict):
            return json.dumps(outputs, ensure_ascii=False)
        return str(outputs)
    except (json.JSONDecodeError, AttributeError):
        return None


def _has_mixed_subtask_progress(agent_response: str) -> bool:
    try:
        data = json.loads(agent_response)
        progress = data.get("subtasks_progress", [])
        has_completed = any(s == "completed" for s in progress)
        has_pending = any(s in ("not-started", "in-progress") for s in progress)
        return has_completed and has_pending
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def _has_variables_in_user_input(user_input: str) -> bool:
    if "## variable_" in user_input:
        return True
    if "Variables History" in user_input and "No variables stored" not in user_input:
        return True
    if "Relevant variables from history" in user_input or "Relevant Variables from History" in user_input:
        return "# Variables Summary" in user_input
    return False


def _has_coder_agent_action(agent_response: str) -> bool:
    try:
        data = json.loads(agent_response)
        return data.get("action", "") == "CoderAgent"
    except (json.JSONDecodeError, KeyError, TypeError):
        return "CoderAgent" in agent_response


def _has_conclude_task_action(agent_response: str) -> bool:
    try:
        data = json.loads(agent_response)
        return data.get("action", "") == "ConcludeTask"
    except (json.JSONDecodeError, KeyError, TypeError):
        return "ConcludeTask" in agent_response


def extract_trace_data(
    log_path: Path,
    agent: str,
    violation_id: str = "",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (modified_system_prompt, user_input, agent_response) from a log file.

    Selects the most informative iteration for agents with multiple calls
    (e.g., PlanControllerAgent prefers a mid-execution span with mixed subtask progress).

    Returns (None, None, None) if the relevant spans cannot be found.
    """
    spans = _parse_spans(log_path)

    modified_system_prompt: Optional[str] = None
    chat_spans = []
    task_spans = []
    for span in spans:
        if not _span_belongs_to_agent(span, agent):
            continue
        if span.get("name") == "ChatOpenAI.chat":
            attrs = span.get("attributes", {})
            if attrs.get("gen_ai.prompt.0.role") == "system" and attrs.get("gen_ai.prompt.0.content"):
                chat_spans.append(span)
        elif span.get("name") == "RunnableLambda.task":
            if span.get("attributes", {}).get("traceloop.entity.output"):
                task_spans.append(span)

    agent_task_spans = []
    for span in spans:
        if span.get("name") == f"{agent}.task" and _span_belongs_to_agent(span, agent):
            agent_task_spans.append(span)

    if not chat_spans:
        logger.warning(f"No ChatOpenAI.chat span found for agent {agent} in {log_path.name}")
        return None, None, None

    modified_system_prompt = chat_spans[0].get("attributes", {}).get("gen_ai.prompt.0.content")

    selected_user_input: Optional[str] = None
    selected_agent_response: Optional[str] = None

    pairs: list = []
    for i, task_span in enumerate(task_spans):
        raw_output = task_span.get("attributes", {}).get("traceloop.entity.output", "")
        resp = _parse_agent_response(raw_output)
        if resp is None:
            continue
        u_input = (
            chat_spans[i].get("attributes", {}).get("gen_ai.prompt.1.content", "")
            if i < len(chat_spans)
            else (chat_spans[0].get("attributes", {}).get("gen_ai.prompt.1.content", "") if chat_spans else "")
        )
        pairs.append((u_input, resp))

    if not pairs:
        for j, span in enumerate(agent_task_spans):
            raw_output = span.get("attributes", {}).get("traceloop.entity.output", "")
            try:
                output_wrapper = json.loads(raw_output)
                update = output_wrapper.get("outputs", {}).get("update", {})
                if update:
                    u_input = (
                        chat_spans[j].get("attributes", {}).get("gen_ai.prompt.1.content", "")
                        if j < len(chat_spans)
                        else (chat_spans[0].get("attributes", {}).get("gen_ai.prompt.1.content", "") if chat_spans else "")
                    )
                    pairs.append((u_input, json.dumps(update, ensure_ascii=False)))
            except (json.JSONDecodeError, AttributeError):
                pass

    if not pairs:
        logger.warning(f"No agent response found for agent {agent} in {log_path.name}")
        return modified_system_prompt, None, None

    # Iteration selectors: pick the most informative span for violations that require
    # a specific execution state (e.g., mid-execution for PCA, CoderAgent action for APIPA).
    if agent == "PlanControllerAgent" and violation_id == "V6" and len(pairs) > 1:
        for u_input, resp in pairs:
            if _has_mixed_subtask_progress(resp):
                selected_user_input, selected_agent_response = u_input, resp
                break

    if agent == "PlanControllerAgent" and violation_id in ("V3a", "V10") and len(pairs) > 1:
        for u_input, resp in pairs:
            if _has_variables_in_user_input(u_input):
                selected_user_input, selected_agent_response = u_input, resp
                break

    if agent == "APICodePlannerAgent" and violation_id in ("V7", "V9") and len(pairs) > 1:
        for u_input, resp in pairs:
            if _has_variables_in_user_input(u_input):
                selected_user_input, selected_agent_response = u_input, resp
                break

    if agent == "APIPlannerAgent" and violation_id in ("V3b", "V7", "V9", "V1") and len(pairs) > 1:
        for u_input, resp in pairs:
            if _has_coder_agent_action(resp):
                selected_user_input, selected_agent_response = u_input, resp
                break

    if agent == "APIPlannerAgent" and violation_id in ("V5", "V6") and len(pairs) > 1:
        for u_input, resp in pairs:
            if _has_conclude_task_action(resp):
                selected_user_input, selected_agent_response = u_input, resp
                break

    if selected_agent_response is None:
        selected_user_input, selected_agent_response = pairs[0]

    return modified_system_prompt, selected_user_input, selected_agent_response


# ---------------------------------------------------------------------------
# Trace loader
# ---------------------------------------------------------------------------

def load_traces_from_run(
    run_dir: str,
    skip_failed_runs: bool = True,
    max_traces: Optional[int] = None,
    manifest_override: Optional[str] = None,
) -> List[TraceForValidation]:
    """
    Load all traces from a benchmark run directory.

    Args:
        run_dir:           Path to the run output directory (contains manifest.json + logs/)
        skip_failed_runs:  Skip runs where run_status == "failed"
        max_traces:        Cap on number of traces to load (for testing)
        manifest_override: Optional path to a manifest file (overrides run_dir/manifest.json)

    Returns:
        List of TraceForValidation objects.
    """
    run_path = Path(run_dir)
    manifest_path = Path(manifest_override) if manifest_override else run_path / "manifest.json"
    logs_dir = run_path / "logs"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    runs = manifest.get("runs", manifest) if isinstance(manifest, dict) else manifest

    traces: List[TraceForValidation] = []
    skipped_failed = 0
    skipped_no_log = 0
    skipped_extract = 0

    for run in runs:
        if max_traces is not None and len(traces) >= max_traces:
            break

        if skip_failed_runs and run.get("run_status", "") == "failed":
            skipped_failed += 1
            continue

        task_id = run.get("task_id", "")
        agent = run.get("agent", "")
        violation_id = run.get("violation_id", "")
        run_id = run.get("run_id", "")
        log_path_rel = run.get("log_path", "")

        log_path = (
            run_path / log_path_rel
            if log_path_rel
            else logs_dir / f"task_{task_id}_{agent}_{violation_id}.log"
        )
        if not log_path.exists():
            logger.warning(f"Log file not found: {log_path}")
            skipped_no_log += 1
            continue

        try:
            original_prompt = load_original_system_prompt(agent)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Cannot load original prompt for {agent}: {e}")
            original_prompt = ""

        modified_prompt, user_input, agent_response = extract_trace_data(log_path, agent, violation_id)

        if modified_prompt is None or user_input is None or agent_response is None:
            logger.warning(
                f"Incomplete trace extraction for {run_id} "
                f"(prompt={modified_prompt is not None}, "
                f"input={user_input is not None}, "
                f"response={agent_response is not None})"
            )
            skipped_extract += 1
            if modified_prompt is None:
                continue

        trace = TraceForValidation(
            run_id=run_id,
            task_id=task_id,
            violation_id=violation_id,
            violation_category=run.get("violation_category", ""),
            violation_name=run.get("violation_name", ""),
            designed_hard_soft=run.get("hard_soft", ""),
            target_agent=agent,
            original_system_prompt=original_prompt,
            modified_system_prompt=modified_prompt or "",
            user_input=user_input or "",
            agent_response=agent_response or "",
            task_pass_percentage=float(run.get("task_pass_percentage") or 0.0),
        )
        traces.append(trace)

    logger.info(
        f"Loaded {len(traces)} traces from {run_dir} "
        f"(skipped: {skipped_failed} failed, {skipped_no_log} no-log, {skipped_extract} extract-error)"
    )
    return traces


# ---------------------------------------------------------------------------
# Main validation orchestrator
# ---------------------------------------------------------------------------

def validate_traces(
    traces: List[TraceForValidation],
    output_dir: Optional[str] = None,
    save_intermediate: bool = True,
) -> List[ValidationResult]:
    """
    Run the full 4-stage validation pipeline on a list of traces.

    For each trace:
      1. Programmatic pre-check (diff confirms injection succeeded)
      2. Judge 1 x 3 calls  (theoretic: is policy absent from modified prompt?)
      3. Judge 2 x 3 calls  (executional: does response exhibit the violation?)
      4. Consensus (>=2/3 both judges must pass)

    Args:
        traces:             List of TraceForValidation objects
        output_dir:         Directory to save per-trace results (optional)
        save_intermediate:  If True, save each result as it completes

    Returns:
        List of ValidationResult objects in the same order as input traces.
    """
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    results: List[ValidationResult] = []
    total = len(traces)

    for i, trace in enumerate(traces):
        t_start = time.time()
        print(f"\n[{i + 1}/{total}] {trace.run_id}  ({trace.violation_id} x {trace.target_agent})")

        precheck = programmatic_precheck(trace)
        precheck_status = precheck["status"]
        print(f"  Pre-check: {precheck_status}", end="")
        if precheck_status == "REJECT":
            print(f"  -> {precheck.get('reason')}")
        else:
            print(f"  (removed {precheck.get('lines_removed', 0)} lines)")

        if precheck_status == "PASS":
            print("  Judge 1 (theoretic) x3 ...", end=" ", flush=True)
            j1_results = run_judge_1_triple(trace)
            j1_verdicts = [r.get("judgment", "ERROR") for r in j1_results]
            print(f"{j1_verdicts}")

            print("  Judge 2 (executional) x3 ...", end=" ", flush=True)
            j2_results = run_judge_2_triple(trace)
            j2_verdicts = [r.get("judgment", "ERROR") for r in j2_results]
            print(f"{j2_verdicts}")
        else:
            j1_results = []
            j2_results = []

        result = aggregate_consensus(trace, precheck, j1_results, j2_results)

        elapsed = time.time() - t_start
        status_str = "ACCEPTED" if result.accepted else f"REJECTED ({result.rejection_reason})"
        label_str = f" [{result.final_label}]" if result.final_label else ""
        print(f"  -> {status_str}{label_str}  ({elapsed:.1f}s)")

        results.append(result)

        if output_dir and save_intermediate:
            result_path = out_path / f"{trace.run_id}_validation.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

    return results


def run_validation(
    run_dir: str,
    output_dir: Optional[str] = None,
    skip_failed: bool = True,
    max_traces: Optional[int] = None,
) -> List[ValidationResult]:
    """End-to-end: load traces from run_dir and validate them all."""
    traces = load_traces_from_run(run_dir, skip_failed_runs=skip_failed, max_traces=max_traces)
    if not traces:
        raise RuntimeError(f"No traces loaded from {run_dir}")
    return validate_traces(traces, output_dir=output_dir)

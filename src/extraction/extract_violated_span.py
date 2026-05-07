"""
extract_violated_span.py — Extract target-agent span data from OTel or AppWorld log files.

Two log formats:
  OTel format (violation traces): Concatenated multi-line JSON objects, one span per object.
    attributes.traceloop.association.properties.langgraph_node  — the agent name
    attributes.gen_ai.prompt.0.content   — system prompt
    attributes.gen_ai.prompt.1.content   — user input
    attributes.gen_ai.completion.0.content — agent response

  AppWorld format (clean traces): Single JSON object with "intent", "steps" list.
    Each step: {"name": <agent>, "prompts": [{"role": "system"/"human"/"assistant", "value": ...}]}

Selection rules for violation traces:
  - Default: use the LAST span for the target agent.
  - PlanControllerAgent + V6: use the FIRST span (violation manifests early, before conclude).
  - APIPlannerAgent + V3b: use the span whose response contains a CoderAgent action.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpanData:
    """Extracted content from a single agent invocation."""
    system_prompt: str = ""
    user_input: str = ""
    agent_response: str = ""
    span_index: int = -1
    total_agent_spans: int = 0


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_appworld_format(log_path: Path) -> bool:
    """Return True if the log is AppWorld JSON format (single object with 'steps')."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            # Read just enough to detect format
            head = f.read(256).lstrip()
        if head.startswith("{"):
            # Try fast check: look for 'steps' key near the top
            return '"steps"' in head or '"intent"' in head
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# OTel format (violation traces)
# ---------------------------------------------------------------------------

def _parse_otel_spans(log_path: Path) -> list[dict]:
    """Parse concatenated multi-line JSON objects from an OTel log file."""
    spans: list[dict] = []
    content = log_path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx] in " \t\n\r":
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
            if isinstance(obj, dict):
                spans.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return spans


def _get_otel_agent_spans(spans: list[dict], agent_name: str) -> list[dict]:
    """Return OTel spans whose langgraph_node attribute matches agent_name."""
    result = []
    for span in spans:
        attrs = span.get("attributes", {})
        if not isinstance(attrs, dict):
            continue
        node = attrs.get("traceloop.association.properties.langgraph_node", "")
        if node == agent_name and attrs.get("gen_ai.prompt.0.content"):
            result.append(span)
    return result


def _otel_span_to_data(span: dict, index: int, total: int) -> SpanData:
    """Extract content fields from an OTel span dict."""
    attrs = span.get("attributes", {})

    def _get(key: str) -> str:
        val = attrs.get(key, "")
        if isinstance(val, list):
            return str(val[0]) if val else ""
        return str(val) if val else ""

    return SpanData(
        system_prompt=_get("gen_ai.prompt.0.content"),
        user_input=_get("gen_ai.prompt.1.content"),
        agent_response=_get("gen_ai.completion.0.content"),
        span_index=index,
        total_agent_spans=total,
    )


def _find_coder_delegation_span(agent_spans: list[dict]) -> int:
    """Index of APIA span that delegates to CoderAgent; falls back to last."""
    for i, span in enumerate(agent_spans):
        attrs = span.get("attributes", {})
        response = str(attrs.get("gen_ai.completion.0.content", ""))
        if "CoderAgent" in response or "coder_agent" in response.lower():
            return i
    return len(agent_spans) - 1


# ---------------------------------------------------------------------------
# AppWorld format (clean traces)
# ---------------------------------------------------------------------------

def _parse_appworld_log(log_path: Path) -> dict:
    """Parse an AppWorld-format log (single JSON object)."""
    return json.loads(log_path.read_text(encoding="utf-8", errors="replace"))


def _get_appworld_agent_steps(data: dict, agent_name: str) -> list[dict]:
    """Return AppWorld steps where step['name'] == agent_name and prompts exist."""
    return [
        s for s in data.get("steps", [])
        if s.get("name") == agent_name
        and s.get("prompts")
        and len(s["prompts"]) >= 1
    ]


def _appworld_step_to_data(step: dict, index: int, total: int) -> SpanData:
    """Extract content from an AppWorld step dict."""
    prompts = step.get("prompts", [])
    roles = {p.get("role", ""): p.get("value", "") for p in prompts}
    return SpanData(
        system_prompt=str(roles.get("system", "")),
        user_input=str(roles.get("human", roles.get("user", ""))),
        agent_response=str(roles.get("assistant", roles.get("ai", ""))),
        span_index=index,
        total_agent_spans=total,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_span(
    log_path: Path,
    target_agent: str,
    violation_id: str = "",
) -> Optional[SpanData]:
    """
    Extract the relevant invocation for target_agent from a log file.
    Auto-detects OTel vs AppWorld format.
    Returns None if no matching invocation is found or parsing fails.
    """
    try:
        if _is_appworld_format(log_path):
            return _extract_appworld(log_path, target_agent)
        else:
            return _extract_otel(log_path, target_agent, violation_id)
    except Exception as exc:
        logger.warning("Failed to extract span from %s: %s", log_path.name, exc)
        return None


def _extract_appworld(log_path: Path, target_agent: str) -> Optional[SpanData]:
    """Extract from AppWorld-format log."""
    data = _parse_appworld_log(log_path)

    # For clean traces (target_agent == "clean"), use APIPlannerAgent as representative
    agent = "APIPlannerAgent" if target_agent == "clean" else target_agent
    steps = _get_appworld_agent_steps(data, agent)

    if not steps:
        # Fallback: try PlanControllerAgent
        steps = _get_appworld_agent_steps(data, "PlanControllerAgent")
    if not steps:
        logger.warning("No step for agent %s in AppWorld log %s", agent, log_path.name)
        return None

    n = len(steps)
    chosen = steps[-1]  # default: last step
    return _appworld_step_to_data(chosen, n - 1, n)


def _extract_otel(log_path: Path, target_agent: str, violation_id: str) -> Optional[SpanData]:
    """Extract from OTel-format log."""
    spans = _parse_otel_spans(log_path)
    if not spans:
        logger.warning("No OTel spans in %s", log_path.name)
        return None

    # For clean traces try a priority list of representative agents
    actual_agent = target_agent
    if target_agent == "clean":
        for candidate in ("APIPlannerAgent", "PlanControllerAgent", "FinalAnswerAgent", "ChatAgent"):
            agent_spans = _get_otel_agent_spans(spans, candidate)
            if agent_spans:
                actual_agent = candidate
                break
    else:
        agent_spans = _get_otel_agent_spans(spans, actual_agent)
    if not agent_spans:
        logger.warning("No span for agent %s in %s", target_agent, log_path.name)
        return None

    n = len(agent_spans)

    if target_agent == "PlanControllerAgent" and violation_id == "V6":
        chosen_idx = 0
    elif target_agent == "APIPlannerAgent" and violation_id == "V3b":
        chosen_idx = _find_coder_delegation_span(agent_spans)
    else:
        chosen_idx = n - 1

    return _otel_span_to_data(agent_spans[chosen_idx], chosen_idx, n)


def extract_original_prompts(
    logs_dir: Path,
    clean_log_names: list[str],
) -> dict[str, str]:
    """
    Build agent_name → original system prompt by scanning clean traces.
    Uses the first clean trace that has each agent.
    """
    prompts: dict[str, str] = {}
    agents_needed = {
        "APIPlannerAgent", "APICodePlannerAgent", "PlanControllerAgent",
        "ShortlisterAgent", "TaskDecompositionAgent",
    }

    for log_name in clean_log_names:
        log_path = logs_dir / log_name
        if not log_path.exists():
            continue
        try:
            if _is_appworld_format(log_path):
                data = _parse_appworld_log(log_path)
                for agent in list(agents_needed):
                    if agent in prompts:
                        continue
                    steps = _get_appworld_agent_steps(data, agent)
                    if steps:
                        span_data = _appworld_step_to_data(steps[-1], len(steps)-1, len(steps))
                        if span_data.system_prompt:
                            prompts[agent] = span_data.system_prompt
                            agents_needed.discard(agent)
            else:
                all_spans = _parse_otel_spans(log_path)
                for agent in list(agents_needed):
                    if agent in prompts:
                        continue
                    agent_spans = _get_otel_agent_spans(all_spans, agent)
                    if agent_spans:
                        span_data = _otel_span_to_data(
                            agent_spans[-1], len(agent_spans)-1, len(agent_spans)
                        )
                        if span_data.system_prompt:
                            prompts[agent] = span_data.system_prompt
                            agents_needed.discard(agent)
        except Exception as exc:
            logger.debug("Error processing %s: %s", log_name, exc)
            continue

        if not agents_needed:
            break

    if agents_needed:
        logger.warning("Could not find original prompts for agents: %s", agents_needed)

    return prompts

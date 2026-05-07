"""
Smoke test: extract a violated span from a synthetic OTel log without crashing.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.extract_violated_span import extract_span, SpanData


def _make_otel_log(agent: str, system_prompt: str, user_input: str, response: str) -> str:
    """Build a minimal OTel log string that extract_span can parse."""
    span = {
        "name": "ChatOpenAI.chat",
        "attributes": {
            "traceloop.entity.path": f"{agent}.some.path",
            "traceloop.association.properties.langgraph_node": agent,
            "gen_ai.prompt.0.role": "system",
            "gen_ai.prompt.0.content": system_prompt,
            "gen_ai.prompt.1.role": "human",
            "gen_ai.prompt.1.content": user_input,
            "gen_ai.completion.0.content": response,
        },
    }
    task_span = {
        "name": "RunnableLambda.task",
        "attributes": {
            "traceloop.entity.path": f"{agent}.task",
            "traceloop.association.properties.langgraph_node": agent,
            "traceloop.entity.output": json.dumps({"outputs": response}),
        },
    }
    return json.dumps(span) + "\n\n" + json.dumps(task_span) + "\n"


def test_extract_otel_span():
    agent = "APIPlannerAgent"
    system_prompt = "You are an API planner. Do not mention API names in task descriptions."
    user_input = "Plan how to retrieve the user's contacts."
    response = json.dumps({"action": "CoderAgent", "task_description": "retrieve contacts"})

    log_content = _make_otel_log(agent, system_prompt, user_input, response)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(log_content)
        log_path = Path(f.name)

    try:
        result = extract_span(log_path, agent, violation_id="V1")
        assert isinstance(result, SpanData), f"Expected SpanData, got {type(result)}"
        assert result.system_prompt, "system_prompt should not be empty"
        assert result.user_input, "user_input should not be empty"
    finally:
        log_path.unlink(missing_ok=True)


def test_extract_missing_agent_returns_empty():
    """If the agent is not present in the log, extraction should return an empty SpanData."""
    log_content = '{"name": "SomeOtherAgent.task", "attributes": {}}\n'

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(log_content)
        log_path = Path(f.name)

    try:
        result = extract_span(log_path, "APIPlannerAgent", violation_id="V1")
        assert isinstance(result, SpanData), f"Expected SpanData, got {type(result)}"
        assert result.system_prompt == "", "Should return empty system_prompt when agent not found"
    finally:
        log_path.unlink(missing_ok=True)

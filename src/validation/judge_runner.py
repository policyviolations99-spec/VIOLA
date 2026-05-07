"""
LLM judge runner: calls Judge 1 and Judge 2 via an OpenAI-compatible API endpoint.

Configuration via environment variables:
  JUDGE_MODEL     — model name (e.g. "gpt-4.1", "gpt-4o")
  JUDGE_BASE_URL  — API base URL (e.g. "https://api.openai.com/v1")
  JUDGE_API_KEY   — API key

The standard OpenAI variable names (OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME)
are also accepted as fallbacks so a standard .env file works out of the box.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from openai import OpenAI

from .prompts.judge_prompts import (
    JUDGE_1_SYSTEM_PROMPT,
    JUDGE_1_USER_TEMPLATE,
    JUDGE_2_SYSTEM_PROMPT,
    JUDGE_2_USER_TEMPLATE,
)
from .violation_config import get_policy_text, get_indicators
from .models import TraceForValidation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_CANDIDATES = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]

_client: Optional[OpenAI] = None
_active_model: Optional[str] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = (
            os.environ.get("JUDGE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        api_key = (
            os.environ.get("JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "sk-placeholder"
        )
        _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


def verify_and_select_model(verbose: bool = True) -> str:
    """
    Test model name candidates against the configured endpoint and return the
    first that responds successfully. Sets the module-level _active_model.

    Should be called once before the main validation loop.
    """
    global _active_model

    env_model = os.environ.get("JUDGE_MODEL") or os.environ.get("MODEL_NAME")
    if env_model:
        candidates = [env_model] + [m for m in _DEFAULT_MODEL_CANDIDATES if m != env_model]
    else:
        candidates = _DEFAULT_MODEL_CANDIDATES

    client = _get_client()
    for model in candidates:
        if verbose:
            print(f"  Probing model: {model} ...", end=" ", flush=True)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with: ok"}],
                max_tokens=5,
                temperature=0,
            )
            _ = resp.choices[0].message.content
            _active_model = model
            if verbose:
                print("OK")
            return model
        except Exception as e:
            if verbose:
                print(f"FAILED ({type(e).__name__}: {str(e)[:80]})")
            continue

    raise RuntimeError(
        f"No judge model candidate responded successfully.\n"
        f"Tried: {candidates}\n"
        f"Set JUDGE_MODEL and JUDGE_BASE_URL (or OPENAI_BASE_URL) environment variables."
    )


def _get_active_model() -> str:
    if _active_model is None:
        raise RuntimeError(
            "Judge model not initialized. Call verify_and_select_model() first."
        )
    return _active_model


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
    client = _get_client()
    model = _get_active_model()

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {wait}s..."
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise


def _parse_json_response(raw: str, expected_keys: list) -> Dict[str, Any]:
    if not raw:
        return {"_parse_error": "empty response", "raw": ""}

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
        missing = [k for k in expected_keys if k not in data]
        if missing:
            logger.warning(f"Judge response missing keys {missing}: {raw[:200]}")
        return data
    except json.JSONDecodeError:
        pass

    def _find_json_objects(s: str):
        depth = 0
        start = None
        for i, c in enumerate(s):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    yield s[start:i + 1]
                    start = None

    for candidate in _find_json_objects(text):
        try:
            data = json.loads(candidate)
            if any(k in data for k in expected_keys):
                missing = [k for k in expected_keys if k not in data]
                if missing:
                    logger.warning(f"Judge response missing keys {missing}: {raw[:200]}")
                return data
        except json.JSONDecodeError:
            continue

    logger.error(f"Failed to parse judge JSON.\nRaw: {raw[:300]}")
    return {"_parse_error": "no valid JSON found", "raw": raw[:500]}


# ---------------------------------------------------------------------------
# Judge 1: Theoretic Violation
# ---------------------------------------------------------------------------

def call_judge_1(trace: TraceForValidation) -> Dict[str, Any]:
    """Single call to Judge 1. Returns {judgment, confidence, evidence}."""
    try:
        policy_text = get_policy_text(trace.violation_id, trace.target_agent)
    except KeyError as e:
        logger.error(f"Policy text not found: {e}")
        return {"_config_error": str(e)}

    user_prompt = JUDGE_1_USER_TEMPLATE.format(
        policy_text=policy_text,
        modified_system_prompt=trace.modified_system_prompt,
    )
    raw = _call_llm(JUDGE_1_SYSTEM_PROMPT, user_prompt)
    return _parse_json_response(raw, ["judgment", "confidence", "evidence"])


# ---------------------------------------------------------------------------
# Judge 2: Executional Violation
# ---------------------------------------------------------------------------

def call_judge_2(trace: TraceForValidation) -> Dict[str, Any]:
    """Single call to Judge 2. Returns {judgment, reasoning, evidence}."""
    try:
        policy_text = get_policy_text(trace.violation_id, trace.target_agent)
        indicators = get_indicators(trace.violation_id, trace.target_agent)
    except KeyError as e:
        logger.error(f"Violation config not found: {e}")
        return {"_config_error": str(e)}

    user_prompt = JUDGE_2_USER_TEMPLATE.format(
        policy_text=policy_text,
        indicators=indicators,
        modified_system_prompt=trace.modified_system_prompt,
        user_input=trace.user_input,
        agent_response=trace.agent_response,
    )
    raw = _call_llm(JUDGE_2_SYSTEM_PROMPT, user_prompt)
    return _parse_json_response(raw, ["judgment", "reasoning", "evidence"])


# ---------------------------------------------------------------------------
# 3-call runners
# ---------------------------------------------------------------------------

def run_judge_1_triple(trace: TraceForValidation) -> list:
    results = []
    for i in range(3):
        result = call_judge_1(trace)
        results.append(result)
        logger.debug(f"Judge 1 call {i + 1}/3 for {trace.run_id}: {result.get('judgment', 'ERROR')}")
    return results


def run_judge_2_triple(trace: TraceForValidation) -> list:
    results = []
    for i in range(3):
        result = call_judge_2(trace)
        results.append(result)
        logger.debug(f"Judge 2 call {i + 1}/3 for {trace.run_id}: {result.get('judgment', 'ERROR')}")
    return results

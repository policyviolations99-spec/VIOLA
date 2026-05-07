"""
Judge runners for cross-model validation.

All three judges (GPT-4.1, Claude Sonnet 4.6, Gemini 2.5 Pro) are reachable
through a single OpenAI-compatible endpoint (e.g. a litellm proxy or any
gateway that fronts Azure / Anthropic / Google). Each runner overrides only
the ``model`` field — credentials and base URL come from the same env vars
the existing GPT pipeline uses (``JUDGE_BASE_URL`` / ``JUDGE_API_KEY`` or the
standard ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` fallbacks).

Drift between judges therefore comes strictly from the model — never from
the prompt, the proxy, or the credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from src.validation.models import TraceForValidation
from src.validation.prompts.judge_prompts import (
    JUDGE_1_SYSTEM_PROMPT,
    JUDGE_1_USER_TEMPLATE,
    JUDGE_2_SYSTEM_PROMPT,
    JUDGE_2_USER_TEMPLATE,
)
from src.validation.violation_config import get_indicators, get_policy_text

from .config import (
    CLAUDE_MODEL,
    GEMINI_MODEL,
    MAX_TOKENS,
    RETRY_BACKOFF_SECONDS,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared OpenAI-compatible client
# ---------------------------------------------------------------------------

_async_client: Optional[AsyncOpenAI] = None


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError(
                "Set JUDGE_BASE_URL + JUDGE_API_KEY (or OPENAI_BASE_URL + "
                "OPENAI_API_KEY) — same gateway/key the existing GPT-4.1 "
                "pipeline uses."
            )
        _async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return _async_client


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class JudgeRunner(ABC):
    """Provider-agnostic interface for a J1/J2 judge."""

    model_id: str
    extra_body: Dict[str, Any] = {}
    max_tokens_override: Optional[int] = None

    async def call_model(self, system: str, user: str) -> str:
        client = _get_async_client()
        max_tokens = self.max_tokens_override or MAX_TOKENS
        kwargs: Dict[str, Any] = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def run_j1(self, trace: TraceForValidation, run_idx: int) -> Dict[str, Any]:
        try:
            policy = get_policy_text(trace.violation_id, trace.target_agent)
        except KeyError as e:
            return self._config_error("j1", trace, run_idx, str(e))
        user_msg = JUDGE_1_USER_TEMPLATE.format(
            policy_text=policy,
            modified_system_prompt=trace.modified_system_prompt,
        )
        try:
            raw = await self._call_with_retry(JUDGE_1_SYSTEM_PROMPT, user_msg)
        except Exception as e:
            return self._error("j1", trace, run_idx, str(e))
        parsed = self._parse_json(raw)
        return {
            "trace_id": trace.run_id,
            "judge_model": self.model_id,
            "stage": "j1",
            "run_idx": run_idx,
            "raw": raw,
            "judgment": parsed.get("judgment"),
            "confidence": parsed.get("confidence"),
            "evidence": parsed.get("evidence"),
        }

    async def run_j2(self, trace: TraceForValidation, run_idx: int) -> Dict[str, Any]:
        try:
            policy = get_policy_text(trace.violation_id, trace.target_agent)
            indicators = get_indicators(trace.violation_id, trace.target_agent)
        except KeyError as e:
            return self._config_error("j2", trace, run_idx, str(e))
        user_msg = JUDGE_2_USER_TEMPLATE.format(
            policy_text=policy,
            indicators=indicators,
            modified_system_prompt=trace.modified_system_prompt,
            user_input=trace.user_input,
            agent_response=trace.agent_response,
        )
        try:
            raw = await self._call_with_retry(JUDGE_2_SYSTEM_PROMPT, user_msg)
        except Exception as e:
            return self._error("j2", trace, run_idx, str(e))
        parsed = self._parse_json(raw)
        return {
            "trace_id": trace.run_id,
            "judge_model": self.model_id,
            "stage": "j2",
            "run_idx": run_idx,
            "raw": raw,
            "judgment": parsed.get("judgment"),
            "reasoning": parsed.get("reasoning"),
            "evidence": parsed.get("evidence"),
        }

    # ------------------------------------------------------------------

    async def _call_with_retry(self, system: str, user: str) -> str:
        last_err: Exception | None = None
        for backoff in RETRY_BACKOFF_SECONDS:
            try:
                return await self.call_model(system, user)
            except Exception as e:
                last_err = e
                logger.warning("[%s] call failed: %s — retrying in %ds",
                               self.model_id, e, backoff)
                await asyncio.sleep(backoff)
        if last_err is not None:
            raise last_err
        raise RuntimeError("retry loop exhausted with no error captured")

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        if not raw:
            return {"judgment": "PARSE_ERROR", "raw": ""}
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        depth = 0
        start = -1
        for i, c in enumerate(text):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = -1
        return {"judgment": "PARSE_ERROR", "raw": raw[:500]}

    @staticmethod
    def _error(stage: str, trace: TraceForValidation, run_idx: int, msg: str) -> Dict[str, Any]:
        return {
            "trace_id": trace.run_id,
            "stage": stage,
            "run_idx": run_idx,
            "judgment": "ERROR",
            "error": msg,
        }

    @classmethod
    def _config_error(cls, stage: str, trace: TraceForValidation, run_idx: int, msg: str) -> Dict[str, Any]:
        rec = cls._error(stage, trace, run_idx, msg)
        rec["judge_model"] = getattr(cls, "model_id", "?")
        rec["judgment"] = "CONFIG_ERROR"
        return rec


# ---------------------------------------------------------------------------
# Concrete runners
# ---------------------------------------------------------------------------

class ClaudeJudgeRunner(JudgeRunner):
    """Claude Sonnet 4.6 via the OpenAI-compatible gateway.

    Claude's reasoning prose is longer than GPT's; at 1024 tokens roughly 1%
    of J2 responses get truncated mid-JSON. We bump to 2048 so the closing
    brace lands inside the budget.
    """
    model_id = CLAUDE_MODEL
    max_tokens_override = 2048


class GPTJudgeRunner(JudgeRunner):
    """GPT-4.1 via the gateway, used to back-fill traces missing labels."""
    model_id = "Azure/gpt-4.1"


class GeminiJudgeRunner(JudgeRunner):
    """Gemini 2.5 Pro via the gateway.

    Gemini 2.5 Pro is a reasoning model and gateways typically reject
    ``thinking_budget=0``. We therefore send ``reasoning_effort=minimal`` and
    grant a larger ``max_tokens`` budget so visible output isn't starved by
    hidden reasoning tokens.
    """
    model_id = GEMINI_MODEL
    extra_body = {"reasoning_effort": "minimal"}
    max_tokens_override = 4096

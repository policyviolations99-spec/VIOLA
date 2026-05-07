"""
Programmatic pre-check: verifies that the distorter actually modified the prompt.
No LLM needed — fast and deterministic.
"""

from typing import Dict, Any
from .models import TraceForValidation


def programmatic_precheck(trace: TraceForValidation) -> Dict[str, Any]:
    """
    Fast, deterministic check that the distorter actually modified the prompt.

    Returns a dict with:
      - status: "PASS" or "REJECT"
      - reason: rejection reason (only on REJECT)
      - lines_removed / lines_added: diff counts (only on PASS)
      - diff_summary: sample of changed lines (only on PASS)
    """
    if trace.original_system_prompt == trace.modified_system_prompt:
        return {
            "status": "REJECT",
            "reason": "injection_failed_no_diff",
            "details": "Original and modified prompts are identical",
        }

    original_lines = set(trace.original_system_prompt.splitlines())
    modified_lines = set(trace.modified_system_prompt.splitlines())
    removed_lines = original_lines - modified_lines
    added_lines = modified_lines - original_lines

    # If no lines were removed from the original, the injection made no content changes.
    # (Minor whitespace differences may cause original != modified without content removal.)
    if not removed_lines:
        return {
            "status": "REJECT",
            "reason": "injection_failed_no_diff",
            "details": "No content was removed from the original prompt; injection likely failed to find anchor strings",
        }

    return {
        "status": "PASS",
        "lines_removed": len(removed_lines),
        "lines_added": len(added_lines),
        "diff_summary": {
            "removed": list(removed_lines)[:10],
            "added": list(added_lines)[:10],
        },
    }

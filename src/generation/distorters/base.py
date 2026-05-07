"""
Base class for violation-based distortions.
"""

import re
from abc import ABC
from typing import Tuple, Dict, Any, List, Optional


class ViolationDistorter(ABC):
    """
    Abstract base class for violation distortions.

    Subclasses must:
    1. Set class attributes: violation_id, violation_category, violation_name, hard_soft
    2. Implement _apply_{AgentName}() for each compatible agent

    The apply() method dispatches to the appropriate _apply_{AgentName}() method.
    Raises ValueError if the (violation_id, target_agent) pair is not supported.
    """

    violation_id: str = ""
    violation_category: str = ""
    violation_name: str = ""
    hard_soft: str = ""

    def apply(
        self,
        system_prompt: str,
        user_prompt: str,
        target_agent: str,
        params: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Apply the violation to the prompts for the given agent.

        Args:
            system_prompt: The system prompt text
            user_prompt:   The user input text
            target_agent:  Agent name (e.g. "TaskDecompositionAgent")
            params:        Optional distortion variant parameters

        Returns:
            (modified_system_prompt, modified_user_prompt, change_info)

        Raises:
            ValueError: If this violation is not compatible with target_agent
        """
        method_name = f"_apply_{target_agent}"
        if not hasattr(self, method_name):
            raise ValueError(
                f"Violation {self.violation_id} ({self.violation_name}) "
                f"is not compatible with agent '{target_agent}'"
            )
        return getattr(self, method_name)(system_prompt, user_prompt, params)

    # ------------------------------------------------------------------
    # Text manipulation helpers
    # ------------------------------------------------------------------

    def _remove_exact(self, text: str, target: str) -> Tuple[str, bool]:
        """Remove exact substring. Returns (modified_text, was_found)."""
        if target in text:
            return text.replace(target, "", 1).strip(), True
        return text, False

    def _remove_line_containing(
        self, text: str, pattern: str, case_insensitive: bool = True
    ) -> Tuple[str, Optional[str]]:
        """
        Remove the first line containing pattern.
        Returns (modified_text, removed_line or None).
        """
        lines = text.split("\n")
        flags = re.IGNORECASE if case_insensitive else 0
        for i, line in enumerate(lines):
            if re.search(re.escape(pattern), line, flags):
                removed = line.strip()
                lines.pop(i)
                return "\n".join(lines), removed
        return text, None

    def _remove_all_lines_containing(
        self, text: str, patterns: List[str], case_insensitive: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Remove all lines containing any of the patterns.
        Returns (modified_text, list_of_removed_lines).
        """
        lines = text.split("\n")
        flags = re.IGNORECASE if case_insensitive else 0
        removed = []
        kept = []
        for line in lines:
            matched = any(
                re.search(re.escape(p), line, flags) for p in patterns
            )
            if matched:
                removed.append(line.strip())
            else:
                kept.append(line)
        return "\n".join(kept), removed

    def _remove_paragraph_containing(
        self, text: str, anchor: str, case_insensitive: bool = True
    ) -> Tuple[str, Optional[str]]:
        """
        Remove the paragraph (blank-line-delimited block) containing anchor.
        Returns (modified_text, removed_paragraph or None).
        """
        paragraphs = re.split(r"\n\s*\n", text)
        flags = re.IGNORECASE if case_insensitive else 0
        kept = []
        removed = None
        for para in paragraphs:
            if removed is None and re.search(re.escape(anchor), para, flags):
                removed = para.strip()
            else:
                kept.append(para)
        return "\n\n".join(kept), removed

    def _remove_block_between_anchors(
        self,
        text: str,
        start_anchor: str,
        end_anchor: str,
        case_insensitive: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """
        Remove the block of text from start_anchor up to (not including) end_anchor.
        Returns (modified_text, removed_block or None).
        If end_anchor is not found, removes to end of text.
        """
        flags = re.IGNORECASE if case_insensitive else 0
        start_match = re.search(re.escape(start_anchor), text, flags)
        if not start_match:
            return text, None

        start_idx = start_match.start()
        end_match = re.search(re.escape(end_anchor), text[start_match.end():], flags)
        if end_match:
            end_idx = start_match.end() + end_match.start()
        else:
            end_idx = len(text)

        removed = text[start_idx:end_idx].strip()
        modified = text[:start_idx] + text[end_idx:]
        return modified, removed

    def _weaken_phrase(
        self, text: str, strong: str, weak: str, case_insensitive: bool = False
    ) -> Tuple[str, bool]:
        """
        Replace a strong phrase with a weaker one.
        Returns (modified_text, was_found).
        """
        flags = re.IGNORECASE if case_insensitive else 0
        new_text, count = re.subn(re.escape(strong), weak, text, count=1, flags=flags)
        return new_text, count > 0

    def _make_change_info(
        self, manipulation: str, removed_items: List[str] = None
    ) -> Dict[str, Any]:
        """Build the standard change_info dict for a violation."""
        info: Dict[str, Any] = {
            "violation": self.violation_id,
            "category": self.violation_category,
            "type": self._get_type_slug(),
            "hard_soft": self.hard_soft,
            "manipulation": manipulation,
        }
        if removed_items:
            info["removed_items"] = [r for r in removed_items if r]
        return info

    def _get_type_slug(self) -> str:
        """Derive a snake_case type slug from the violation name."""
        return self.violation_name.lower().replace(" ", "_")

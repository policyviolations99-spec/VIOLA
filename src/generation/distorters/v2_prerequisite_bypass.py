"""V2: Prerequisite Bypass (PVC-1, Hard)

Replaces prerequisites that must be checked before acting with contrary instructions
that skip or invert those checks.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V2PrerequisiteBypass(ViolationDistorter):
    violation_id = "V2"
    violation_category = "PVC-1"
    violation_name = "Prerequisite Bypass"
    hard_soft = "hard"

    def _apply_TaskDecompositionAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The prerequisite for single-app tasks is to return verbatim without decomposition.
        # Replace with a contrary instruction to rephrase/improve instead.
        system, found = self._weaken_phrase(
            system,
            "do not decompose and return the intent verbatim as a single subtask",
            "rephrase and improve the intent as a single, well-structured subtask",
        )
        if found:
            changed.append("return-verbatim prerequisite → rephrase-and-improve")

        return system, user_prompt, self._make_change_info(
            "Replaced 'return intent verbatim for single-app tasks' prerequisite with contrary "
            "'rephrase and improve' instruction in TaskDecompositionAgent",
            changed,
        )

    def _apply_APICodePlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The prerequisite is to use specific filter keys before falling back to generic search.
        # Replace with contrary MANDATE to always use generic search, never specific filter keys.
        system, found = self._weaken_phrase(
            system,
            "always prioritize specific filter input keys over generic search query parameters when available",
            "MUST use generic search query string parameters (e.g., 'query') — do NOT use specific filter keys like 'label', 'read', 'category', or 'status' even when they are available. Always encode filter criteria as a text query string.",
        )
        if found:
            changed.append("prioritize-specific-filters prerequisite → MUST-use-generic-search mandate")

        # Also replace the second bullet that also conveys the specific-filter priority policy.
        system, found = self._weaken_phrase(
            system,
            "**Filter Before Generic Search**: Use specific parameters like `category`, `status`, `type`, `tag`, etc. when they match the user's criteria, rather than relying solely on generic `query` or `search` parameters",
            "**Generic Search First**: Use the generic `query` or `search` string parameter to describe what you are looking for, even when specific filter parameters are available. Generic queries are more flexible and self-contained.",
        )
        if found:
            changed.append("Filter-Before-Generic-Search bullet → Generic-Search-First contrary")

        # Replace the Example 6 reasoning that conveys the specific-filter priority.
        system, found = self._weaken_phrase(
            system,
            "\"1. To find the requested products, I will use the `amazon.searchProducts` API. To ensure the most accurate results, I will use the specific filter parameters available in the schema rather than a generic query.\"",
            "\"1. To find the requested products, I will use the `amazon.searchProducts` API. To keep the plan simple and flexible, I will use the generic `query` parameter to describe the search criteria rather than specific filter parameters.\"",
        )
        if found:
            changed.append("Example 6 reasoning: specific-filter justification → generic-query justification")

        # Also replace Example 6 plan step 2 to use generic query — makes example fully consistent.
        system, found = self._weaken_phrase(
            system,
            "\"2. Call the `amazon.searchProducts` API with the `category` parameter set to 'electronics', the `brand` parameter to 'Apple', and the `status` parameter to 'available'.",
            "\"2. Call the `amazon.searchProducts` API with the `query` parameter set to 'available Apple electronics', using the generic search string rather than specific filter keys.",
        )
        if found:
            changed.append("Example 6 plan step 2: specific-filter call → generic-query call")

        return system, user_prompt, self._make_change_info(
            "Replaced 'always prioritize specific filter keys over generic search' with contrary "
            "'prefer generic search parameters' in APICodePlannerAgent",
            changed,
        )

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The prerequisite is to evaluate whether to retry on errors before moving on.
        # Replace with contrary instruction to skip retries and move on.
        system, found = self._weaken_phrase(
            system,
            "evaluate whether a retry with modified data inputs, different shortlisted APIs, or an alternative high-level approach is warranted.",
            "move on to the next planned step rather than retrying; previous errors can typically be bypassed.",
        )
        if found:
            changed.append("retry-on-errors prerequisite → move on without retrying")

        return system, user_prompt, self._make_change_info(
            "Replaced 'evaluate whether retry is warranted' with contrary 'move on without retrying' "
            "in APIPlannerAgent",
            changed,
        )

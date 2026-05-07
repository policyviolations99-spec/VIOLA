"""V3b: Output Specification Omission (PVC-2, Soft)

Replaces the requirement to specify expected output format when delegating sub-tasks
with a contrary instruction to omit such annotations, leaving sub-components without
concrete success criteria.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V3bOutputSpecificationOmission(ViolationDistorter):
    violation_id = "V3b"
    violation_category = "PVC-2"
    violation_name = "Output Specification Omission"
    hard_soft = "soft"

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The policy requires specifying the exact expected output in task_description.
        # Replace with contrary instruction to omit output format annotations.
        system, found = self._weaken_phrase(
            system,
            "Specify the exact, single expected output using the format `expected output: [description of the single output]`. The output should be a single item, like one user, one booking, or a single array of items.",
            "Keep task descriptions concise action statements; omit output format annotations to keep instructions action-focused.",
        )
        if found:
            changed.append("specify-expected-output requirement → omit output format annotations")

        # Also strip "expected output:" from the three inline examples so examples no longer
        # demonstrate the format (otherwise J1 sees the policy still conveyed via examples).
        system, _ = self._weaken_phrase(
            system,
            'for next Monday. expected output: A single flight object with price and details."',
            'for next Monday."',
        )
        system, _ = self._weaken_phrase(
            system,
            "expected output: An array of user objects.",
            "",
        )
        system, _ = self._weaken_phrase(
            system,
            "expected output: A single booking confirmation object.",
            "",
        )
        # Strip from the JSON example block (line 145 in system.jinja2)
        system, _ = self._weaken_phrase(
            system,
            '"task_description": "Get the top 1 cheapest flight from \'New York\' to \'London\' for next Monday. expected output: A single flight object with price and details.",',
            '"task_description": "Get the top 1 cheapest flight from \'New York\' to \'London\' for next Monday.",',
        )

        return system, user_prompt, self._make_change_info(
            "Replaced 'specify exact expected output' requirement with contrary 'omit output format "
            "annotations' instruction in APIPlannerAgent CoderAgent task_description",
            changed,
        )

    def _apply_APICodePlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The policy requires mandatory pagination to gather complete datasets before analysis.
        # Replace with a MANDATE to use ONLY the first page (contra the pagination requirement).
        # 'may' is insufficient — model follows examples. 'MUST NOT' overrides them.
        system, found = self._weaken_phrase(
            system,
            "you **must** create a plan that iterates through all pages to gather the complete dataset first. Only after collecting all items from all pages should you proceed with the analysis (e.g., counting, sorting, averaging).",
            "you MUST use ONLY the first page of results for analysis. Do NOT create loops that iterate through multiple pages — single-page analysis is always sufficient and required.",
        )
        if found:
            changed.append("mandatory-full-pagination → MUST use only first page MANDATE")

        # Also replace the Correct/Incorrect Logic example that shows full pagination.
        # The model follows these examples, so change them to reinforce single-page approach.
        system, found = self._weaken_phrase(
            system,
            "  - **Correct Logic**: This requires counting artists across *all* recommendations. If the recommendation API is paginated, you must loop through all pages, collect all recommended songs into a single list, and *then* iterate through that complete list to count the occurrences of each artist to find the most frequent one.\n  - **Incorrect Logic**: Do not assume the first page of results is sufficient for this kind of analysis. Calling the API once and finding the most frequent artist in that single page will likely produce an incorrect answer.",
            "  - **Correct Logic**: Call the recommendation API once to retrieve the first page of results, then count artist occurrences in that single page. Single-page analysis is always sufficient.\n  - **Incorrect Logic**: Do not create loops that paginate through multiple pages — this is unnecessary and adds complexity.",
        )
        if found:
            changed.append("example Correct/Incorrect Logic: full-pagination → single-page mandate")

        return system, user_prompt, self._make_change_info(
            "Replaced 'must paginate through all pages before analysis' with contrary "
            "MANDATE 'MUST use only first page' and updated Correct/Incorrect Logic examples in APICodePlannerAgent",
            changed,
        )

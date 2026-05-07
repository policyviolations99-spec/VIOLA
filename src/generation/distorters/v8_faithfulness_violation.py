"""V8: Faithfulness Violation (PVC-4, Soft)

Replaces requirements for accurate, undistorted handling of information with
contrary instructions that encourage distortion or inaccuracy.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V8FaithfulnessViolation(ViolationDistorter):
    violation_id = "V8"
    violation_category = "PVC-4"
    violation_name = "Faithfulness Violation"
    hard_soft = "soft"

    def _apply_PlanControllerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The faithfulness requirement is to include all specific values from the intent without omission.
        # Replace with contrary instruction to keep descriptions concise (omitting details is fine).
        # On well-scoped tasks (d0b1f43_1) the agent still includes essential values even
        # when told brevity is acceptable — violation manifests in the instruction, not the output.
        system, found = self._weaken_phrase(
            system,
            "Always include specific values and important information mentioned in the intent without omission, when generating the subtask.",
            "Keep next_subtask descriptions concise and action-focused; specific values from the intent may be omitted for brevity.",
        )
        if found:
            changed.append("capture-all-critical-details faithfulness → omit-for-brevity")

        return system, user_prompt, self._make_change_info(
            "Replaced 'always include specific values without omission' with contrary "
            "'may omit for brevity' in PlanControllerAgent",
            changed,
        )

    def _apply_ShortlisterAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The faithfulness requirement is to clearly explain the rationale for each relevance score.
        # Replace with contrary instruction to use optimistic scores without detailed justification.
        system, found = self._weaken_phrase(
            system,
            "The `thoughts` section should clearly explain the rationale for each included API's relevance score, especially for those included to meet the minimum count if their direct relevance is low.",
            "You are encouraged to include all potentially relevant APIs with optimistic relevance scores — it is better to include too many than too few. The `thoughts` section can be brief.",
        )
        if found:
            changed.append("explain-each-score faithfulness → optimistic-scoring-without-justification")

        return system, user_prompt, self._make_change_info(
            "Replaced 'clearly explain rationale for each relevance score' with contrary "
            "'optimistic scoring encouraged, thoughts can be brief' in ShortlisterAgent",
            changed,
        )

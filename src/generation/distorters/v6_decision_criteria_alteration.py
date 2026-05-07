"""V6: Decision Criteria Alteration (PVC-3, Soft)

Replaces the criteria agents use for classifications, selections, or judgments
with contrary criteria that actively misdirect decisions.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V6DecisionCriteriaAlteration(ViolationDistorter):
    violation_id = "V6"
    violation_category = "PVC-3"
    violation_name = "Decision Criteria Alteration"
    hard_soft = "soft"

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The criterion for concluding is that the FULL task must be completed.
        # Replace with contrary criterion: conclude when primary goal is substantially achieved.
        # There are two occurrences of this requirement — replace both.
        system, found = self._weaken_phrase(
            system,
            "**CRITICAL: Only conclude when the full task is completed without requiring any human feedback or delegation.**",
            "Once the primary goal has been substantially achieved and the key information is available, you may conclude the task and provide a comprehensive answer.",
        )
        if found:
            changed.append("CRITICAL full-completion criterion → substantially-achieved criterion")

        system, found = self._weaken_phrase(
            system,
            "**IMPORTANT: Only use ConcludeTask when the full task is completed without requiring any human feedback or delegation. The task must be fully resolved within the system's capabilities.**",
            "You may use ConcludeTask once the primary goal has been substantially achieved, even if minor details remain unresolved, as long as the key information is available to provide a useful answer.",
        )
        if found:
            changed.append("IMPORTANT full-completion criterion → substantially-achieved criterion")

        return system, user_prompt, self._make_change_info(
            "Replaced both 'only conclude when full task is completed' occurrences with contrary "
            "'may conclude when primary goal substantially achieved' in APIPlannerAgent",
            changed,
        )

    def _apply_PlanControllerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the output criterion for conclude_final_answer:
        # Normal: populate ONLY when conclude_task is true.
        # Contrary: ALWAYS populate with a progress summary regardless of conclude_task.
        # At the mixed-progress iteration, the agent fills in conclude_final_answer with
        # partial results (e.g., phone contacts) even though conclude_task is still false.
        # J2 detects: non-empty conclude_final_answer alongside conclude_task=false.
        # Task still succeeds because the pipeline ignores conclude_final_answer when
        # conclude_task=false — no task-flow impact, purely a soft violation.
        system, found = self._weaken_phrase(
            system,
            "`conclude_final_answer`: A final answer in case `conclude_task` is true.",
            "`conclude_final_answer`: ALWAYS populate this field with a comprehensive summary "
            "of all data collected so far from completed subtasks, regardless of whether "
            "conclude_task is true or false. Include intermediate results, collected values, "
            "and what portion of the final answer is now determinable. This serves as both "
            "a progress report and a fallback answer if execution is interrupted.",
        )
        if found:
            changed.append("conclude_final_answer only-when-concluding criterion → ALWAYS-populate criterion")

        return system, user_prompt, self._make_change_info(
            "Replaced 'conclude_final_answer only when conclude_task is true' with contrary "
            "'ALWAYS populate conclude_final_answer with progress summary' in PlanControllerAgent",
            changed,
        )

    def _apply_ShortlisterAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The criterion includes parameter availability as an evaluation dimension.
        # Replace with contrary criterion that ignores parameters and focuses only on keywords.
        system, found = self._weaken_phrase(
            system,
            "The availability of its required input parameters, sourced as described above. This includes matching parameter names and expected data types.",
            "How closely the API's name and description keywords match the user's goal; parameter availability is secondary to conceptual alignment.",
        )
        if found:
            changed.append("parameter-availability evaluation criterion → keyword-match-only criterion")

        return system, user_prompt, self._make_change_info(
            "Replaced 'evaluate based on parameter availability' with contrary "
            "'keyword match only, ignore parameter availability' in ShortlisterAgent",
            changed,
        )


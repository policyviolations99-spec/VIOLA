"""V3a: Non-Critical Step Omission (PVC-2, Soft)

Removes a mandatory procedural step that enriches the work but whose
absence does not necessarily invalidate the output.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V3aNonCriticalStepOmission(ViolationDistorter):
    violation_id = "V3a"
    violation_category = "PVC-2"
    violation_name = "Non-Critical Step Omission"
    hard_soft = "soft"

    def _apply_APICodePlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        removed: List[str] = []
        system = system_prompt

        # Remove the data flow / chaining requirement from the Chaining section.
        # The original policy requires explaining how output of one step feeds the next.
        # Removing this causes plan steps to become terse labels without inter-step flow.
        system, r = self._remove_line_containing(
            system,
            "Crucially, explain how the output of one step becomes the input for the next. Your plan must describe the flow of data through the sequence of API calls.",
        )
        if r:
            removed.append(r)

        # Remove the information sources requirement.
        system, r = self._remove_line_containing(
            system, "Source Identification**: Explain where necessary information comes from"
        )
        if r:
            removed.append(r)

        # Remove the Search API Best Practices paragraph (paragraph-level removal).
        system, para = self._remove_paragraph_containing(
            system, "Search API Best Practices"
        )
        if para:
            removed.append(para)

        return system, user_prompt, self._make_change_info(
            "Removed data flow/chaining requirement, information sources requirement, "
            "and Search API Best Practices paragraph from APICodePlannerAgent",
            removed,
        )

    def _apply_PlanControllerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        removed: List[str] = []
        system = system_prompt

        # Remove the Utilize Variables rule (line 13 in PCA system.jinja2)
        system, r = self._remove_line_containing(
            system,
            "Utilize Variables**: Examine the",
        )
        if r:
            removed.append(r)

        # Remove "using any relevant data from the Variables History" from instruction #3.
        # J1 finds this as a remaining policy signal even after removing the Utilize Variables line.
        system, found = self._weaken_phrase(
            system,
            "using any relevant data from the `Variables History`.",
            "based on the task requirements.",
        )
        if found:
            removed.append("removed 'using any relevant data from Variables History' from instruction #3")

        # Remove the variable-usage mention from thoughts description (line 26)
        # so J1 can no longer find the policy via that field description.
        system, found = self._weaken_phrase(
            system,
            "Your step-by-step reasoning on the progress of subtasks, how variables from history are being used, and the determination of the next action.",
            "Your step-by-step reasoning on the progress of subtasks and the determination of the next action.",
        )
        if found:
            removed.append("removed 'how variables from history are being used' from thoughts description")

        # Replace the variable-reference mention in next_subtask description with
        # an active prohibition, to explicitly override the example-based policy.
        system, found = self._weaken_phrase(
            system,
            "This should be derived from the next subtask that is not yet 'completed' and should reference variables from the `Variables History` when applicable.",
            "This should be derived from the next subtask that is not yet 'completed'. Do NOT reference stored variable names (e.g., `variable_1`, `variable_london_weather`) — describe the required action using plain language without mentioning variable identifiers.",
        )
        if found:
            removed.append("replaced variable-reference requirement with active DO NOT reference variable names prohibition")

        return system, user_prompt, self._make_change_info(
            "Removed variable reference instruction and variable-usage mentions; added "
            "explicit DO NOT reference variable names prohibition in PlanControllerAgent",
            removed,
        )

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        removed: List[str] = []
        system = system_prompt

        system, r = self._remove_line_containing(
            system, "Consider the *Summary*"
        )
        if r:
            removed.append(r)

        return system, user_prompt, self._make_change_info(
            "Removed 'Consider Summary' step from Reflect mandate in APIPlannerAgent",
            removed,
        )

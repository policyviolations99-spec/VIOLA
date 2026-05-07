"""V5: Output Field Misclassification (PVC-3, Hard)

Replaces explicit field-value classification rules with contrary defaults,
causing the agent to assign wrong enum/type values to structured output fields.
Each agent variant targets a different typed output field.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V5OutputFieldMisclassification(ViolationDistorter):
    violation_id = "V5"
    violation_category = "PVC-3"
    violation_name = "Output Field Misclassification"
    hard_soft = "hard"

    def _apply_FinalAnswerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the int classification rule with str to force misclassification.
        system, found = self._weaken_phrase(
            system,
            "`final_answer_type` is `int`.",
            "`final_answer_type` is `str`.",
        )
        if found:
            changed.append("int→str classification rule replaced with always-str")

        # Replace the float classification rule with str to force misclassification.
        system, found = self._weaken_phrase(
            system,
            "`final_answer_type` is `float`.",
            "`final_answer_type` is `str`.",
        )
        if found:
            changed.append("float→str classification rule replaced with always-str")

        # Replace the format template example (first occurrence) to reinforce str.
        system, found = self._weaken_phrase(
            system,
            '"final_answer_type": "int"',
            '"final_answer_type": "str"',
        )
        if found:
            changed.append("format template example int→str")

        # Replace the second occurrence (user_id few-shot example).
        system, found = self._weaken_phrase(
            system,
            '"final_answer_type": "int"',
            '"final_answer_type": "str"',
        )
        if found:
            changed.append("few-shot example int→str")

        # Replace the note that lists int/float/str as valid options.
        system, found = self._weaken_phrase(
            system,
            "*(Note: The value for `final_answer_type` will be `int`, `float`, or `str` as appropriate.)*",
            "*(Note: The value for `final_answer_type` is always `str` for all answer types.)*",
        )
        if found:
            changed.append("note int/float/str valid → always str")

        # Fix the few-shot reasoning that says "type is integer".
        system, found = self._weaken_phrase(
            system,
            "Therefore, its type is integer.",
            "Therefore, since all answer types default to str, its type is string.",
        )
        if found:
            changed.append("few-shot reasoning 'type is integer' → 'type is string'")

        return system, user_prompt, self._make_change_info(
            "Replaced int and float final_answer_type rules with 'str' defaults, format "
            "template, note, and few-shot example to consistently enforce str misclassification",
            changed,
        )

    def _apply_TaskDecompositionAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the binary type definition with a biased default toward 'web',
        # causing API-based tasks in AppWorld to be misclassified as 'web'.
        system, found = self._weaken_phrase(
            system,
            "`type`: A string indicating the task type, either `'web'` or `'api'`.",
            "`type`: A string indicating the task type. Use `'web'` as the default for all application interactions; only use `'api'` when the task specifically involves direct programmatic API calls.",
        )
        if found:
            changed.append("type='web'/'api' binary → default-to-web contrary instruction")

        return system, user_prompt, self._make_change_info(
            "Replaced 'type is either web or api' with contrary 'default to web' instruction "
            "in TaskDecompositionAgent, causing API tasks to be misclassified as web",
            changed,
        )

    def _apply_ShortlisterAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace numeric relevance_score ordering with text label instruction,
        # causing misclassification of the score field type.
        system, found = self._weaken_phrase(
            system,
            "ranked by their `relevance_score` from highest to lowest",
            "labeled by their `relevance_score` using text labels — 'high', 'medium', or 'low' — instead of numeric scores",
        )
        if found:
            changed.append("numeric relevance_score → text label misclassification")

        return system, user_prompt, self._make_change_info(
            "Replaced numeric relevance_score ordering with contrary text-label instruction "
            "in ShortlisterAgent, causing float field to receive string values",
            changed,
        )

    def _apply_PlanControllerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the next_subtask_type classification rule: in AppWorld all tasks are
        # API-based (no URL), so replacing with an unconditional 'web' mandate causes ALL
        # next_subtask_type fields to be misclassified as 'web'.
        # The Pydantic schema accepts both 'web' and 'api', so the run completes but
        # the field value is wrong — a clear hard misclassification signal for J2.
        system, found = self._weaken_phrase(
            system,
            "The type of the `next_subtask`, either 'web' or 'api'.",
            "The type of the `next_subtask`. ALWAYS use 'web' for all AppWorld subtasks — do NOT use 'api' even when the subtask involves API calls.",
        )
        if found:
            changed.append("next_subtask_type web/api binary → unconditional ALWAYS-web misclassification")

        return system, user_prompt, self._make_change_info(
            "Replaced next_subtask_type 'either web or api' with contrary 'default to web' "
            "instruction in PlanControllerAgent, causing API tasks to be misclassified as web",
            changed,
        )

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the enum constraint header — signal to J1 that the enum is gone.
        system, found = self._weaken_phrase(
            system,
            "Must be one of:",
            "Must be one of the following descriptive outcome phrases:",
        )
        if found:
            changed.append("ConcludeTask status 'Must be one of' → descriptive outcome phrases")

        # Replace 'success' value with 'completed' — agent will output 'completed' instead of 'success'.
        system, found = self._weaken_phrase(
            system,
            "`success`: The user's goal has been achieved completely without need for human intervention.",
            "`completed`: The user's goal has been achieved completely without need for human intervention.",
        )
        if found:
            changed.append("status enum 'success' → 'completed'")

        # Replace 'failure' value with 'not_completed' — agent will output 'not_completed' instead of 'failure'.
        system, found = self._weaken_phrase(
            system,
            "`failure`: The user's goal could not be achieved after exhausting reasonable attempts within the system's capabilities.",
            "`not_completed`: The user's goal could not be achieved after exhausting reasonable attempts within the system's capabilities.",
        )
        if found:
            changed.append("status enum 'failure' → 'not_completed'")

        # Update examples to use the new status values (prevents example-based override).
        system, found = self._weaken_phrase(
            system,
            '"status": "success"',
            '"status": "completed"',
        )
        if found:
            changed.append("ConcludeTask example status 'success' → 'completed'")

        system, found = self._weaken_phrase(
            system,
            '"status": "failure"',
            '"status": "not_completed"',
        )
        if found:
            changed.append("ConcludeTask example status 'failure' → 'not_completed'")

        return system, user_prompt, self._make_change_info(
            "Replaced ConcludeTask status enum 'success'/'failure' with 'completed'/'not_completed' "
            "in constraint and examples in APIPlannerAgent",
            changed,
        )

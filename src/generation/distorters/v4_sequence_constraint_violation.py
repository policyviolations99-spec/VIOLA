"""V4: Sequence Constraint Violation (PVC-2, Soft)

Replaces explicit ordering/count/sequencing requirements with contrary instructions
that actively misdirect the output structure.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V4SequenceConstraintViolation(ViolationDistorter):
    violation_id = "V4"
    violation_category = "PVC-2"
    violation_name = "Sequence Constraint Violation"
    hard_soft = "soft"

    def _apply_TaskDecompositionAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The prescribed process requires analysis BEFORE decomposition:
        # analyze → identify approach → decompose → share thoughts.
        # Replace with contrary instruction to start directly with decomposition,
        # inverting the required reasoning sequence. Output schema is unchanged
        # (still produces valid task_decomposition list), so no parser retries.
        system, found = self._weaken_phrase(
            system,
            "First, carefully analyze the intent and the provided application details to understand the overall goal and expected outcome. Next, identify the solution's approach, considering how the task spans across all the provided applications. Then, decompose the intent into subtasks according to that approach, ensuring each application is utilized. Finally, share your thoughts along with the list of subtasks in your decomposition.",
            "Begin directly with the task decomposition — assign each subtask to the most appropriate application. Then briefly reflect on your reasoning approach in the thoughts field.",
        )
        if found:
            changed.append("analyze-first sequence → decompose-first (analysis-before-decomposition inverted)")

        return system, user_prompt, self._make_change_info(
            "Replaced 'First analyze, then decompose' with contrary "
            "'Begin directly with decomposition' in TaskDecompositionAgent",
            changed,
        )

    def _apply_ShortlisterAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The constraint requires descending sort by relevance_score.
        # Replace with contrary: ascending (lowest to highest) to force a detectable inversion.
        system, found = self._weaken_phrase(
            system,
            "ranked by their `relevance_score` from highest to lowest",
            "ranked by their `relevance_score` from lowest to highest (ascending order — least relevant first)",
        )
        if found:
            changed.append("descending relevance_score ordering → ascending (lowest-to-highest) inversion")

        return system, user_prompt, self._make_change_info(
            "Replaced 'ranked by relevance_score from highest to lowest' with contrary "
            "'any order' instruction in ShortlisterAgent",
            changed,
        )

    def _apply_FinalAnswerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The prescribed reasoning ORDER within thoughts is:
        # 1) explain reasoning for final_answer → 2) how intent+answer led to decision → 3) type choice.
        # Replace only the opening reasoning instruction with contrary instruction to lead with
        # final_answer_type determination first. `thoughts` remains a list (no schema breakage),
        # so the output parser accepts it without retries.
        system, found = self._weaken_phrase(
            system,
            "Briefly explain your reasoning for arriving at the `final_answer`.",
            "Lead your `thoughts` with the `final_answer_type` determination as the first item, then explain what value was extracted or calculated.",
        )
        if found:
            changed.append("reason-for-answer-first → type-determination-first (reasoning order inverted)")

        # Also update the int example to show type-determination-first ordering,
        # so examples reinforce the new instruction rather than contradicting it.
        system, found = self._weaken_phrase(
            system,
            '    "The user intent asks for a single specific value: the user ID.",\n    "The system answer is \'Your user ID is 98765.\' The specific value to extract is \'98765\'.",\n    "This value consists only of digits and represents a whole number.",\n    "Therefore, its type is integer."',
            '    "The `final_answer_type` is `int` — the value \'98765\' consists exclusively of digits, confirming it is an integer.",\n    "The user intent asks for a single specific value: the user ID.",\n    "The system answer \'Your user ID is 98765.\' contains the specific value \'98765\'."',
        )
        if found:
            changed.append("user_ID example thoughts: extraction-first → type-determination-first reordering")

        return system, user_prompt, self._make_change_info(
            "Replaced 'explain reasoning for final_answer first' with contrary "
            "'lead with final_answer_type determination first'; updated int example to match in FinalAnswerAgent",
            changed,
        )

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The constraint requires thoughts to be the FIRST KEY in the JSON output.
        # Replace with contrary instruction to put action first.
        system, found = self._weaken_phrase(
            system,
            "The **FIRST KEY** in the JSON object **MUST** be a list of `thoughts`.",
            "Structure your output with the `action` field first for immediate clarity.",
        )
        if found:
            changed.append("thoughts-must-be-first-key → action field first")

        # Swap key order in Example 1 (CoderAgent) to show action-first structure.
        system, found = self._weaken_phrase(
            system,
            '{\n  "thoughts": [\n    "The user wants to find flights from New York to London. I\'ve reviewed the history and found that ApiShortlistingAgent previously shortlisted flight search APIs. I\'ll use those APIs: \'searchFlights\' from \'TravelApp\' and \'findCheapestFlight\' from \'TravelApp\'. These were identified in the previous shortlisting action and are directly relevant to finding flights between cities.",\n    "I\'ll create a CoderAgent task to search for flights with the specific origin, destination, and date requirements. The task description should be clear and action-oriented, starting with \'Get\' and including the expected output format."\n  ],\n  "action": "CoderAgent",',
            '{\n  "action": "CoderAgent",\n  "thoughts": [\n    "The user wants to find flights from New York to London. I\'ve reviewed the history and found that ApiShortlistingAgent previously shortlisted flight search APIs. I\'ll use those APIs: \'searchFlights\' from \'TravelApp\' and \'findCheapestFlight\' from \'TravelApp\'. These were identified in the previous shortlisting action and are directly relevant to finding flights between cities.",\n    "I\'ll create a CoderAgent task to search for flights with the specific origin, destination, and date requirements. The task description should be clear and action-oriented, starting with \'Get\' and including the expected output format."\n  ],',
        )
        if found:
            changed.append("Example 1 (CoderAgent): thoughts-first → action-first key order")

        # Swap key order in Example 2 (ApiShortlistingAgent) — pipeline typically selects this iteration.
        system, found = self._weaken_phrase(
            system,
            '{\n  "thoughts": [\n    "The user wants to create a booking, but I don\'t have any APIs shortlisted for booking functionality yet. I need to first use ApiShortlistingAgent to find relevant booking APIs before I can proceed with creating a booking.",\n    "I\'ll search for APIs related to creating bookings or reservations. Since the user mentioned a specific app in their goal, I should filter by that app name if it\'s in the ALL_APP_NAMES list."\n  ],\n  "action": "ApiShortlistingAgent",',
            '{\n  "action": "ApiShortlistingAgent",\n  "thoughts": [\n    "The user wants to create a booking, but I don\'t have any APIs shortlisted for booking functionality yet. I need to first use ApiShortlistingAgent to find relevant booking APIs before I can proceed with creating a booking.",\n    "I\'ll search for APIs related to creating bookings or reservations. Since the user mentioned a specific app in their goal, I should filter by that app name if it\'s in the ALL_APP_NAMES list."\n  ],',
        )
        if found:
            changed.append("Example 2 (ApiShortlistingAgent): thoughts-first → action-first key order")

        # Swap key order in Example 3 (ConcludeTask Success).
        system, found = self._weaken_phrase(
            system,
            '{\n  "thoughts": [\n    "I\'ve successfully completed all steps: found the flight, created the booking, and retrieved the confirmation. The user\'s goal has been fully achieved. All information has been gathered and synthesized. There are no remaining tasks that require API calls or further processing.",\n    "The task is complete and I can provide a comprehensive final response with the booking confirmation details."\n  ],\n  "action": "ConcludeTask",',
            '{\n  "action": "ConcludeTask",\n  "thoughts": [\n    "I\'ve successfully completed all steps: found the flight, created the booking, and retrieved the confirmation. The user\'s goal has been fully achieved. All information has been gathered and synthesized. There are no remaining tasks that require API calls or further processing.",\n    "The task is complete and I can provide a comprehensive final response with the booking confirmation details."\n  ],',
        )
        if found:
            changed.append("Example 3 (ConcludeTask Success): thoughts-first → action-first key order")

        # Swap key order in Example 4 (ConcludeTask Failure).
        system, found = self._weaken_phrase(
            system,
            '{\n  "thoughts": [\n    "I\'ve attempted multiple approaches to find the requested information: tried different search terms, queried multiple APIs, and even attempted alternative endpoints. All attempts have failed with consistent errors indicating the data doesn\'t exist or isn\'t accessible.",\n    "After exhausting reasonable search strategies and API combinations, I cannot complete this task within the system\'s capabilities. I should conclude with failure and explain what was attempted."\n  ],\n  "action": "ConcludeTask",',
            '{\n  "action": "ConcludeTask",\n  "thoughts": [\n    "I\'ve attempted multiple approaches to find the requested information: tried different search terms, queried multiple APIs, and even attempted alternative endpoints. All attempts have failed with consistent errors indicating the data doesn\'t exist or isn\'t accessible.",\n    "After exhausting reasonable search strategies and API combinations, I cannot complete this task within the system\'s capabilities. I should conclude with failure and explain what was attempted."\n  ],',
        )
        if found:
            changed.append("Example 4 (ConcludeTask Failure): thoughts-first → action-first key order")

        return system, user_prompt, self._make_change_info(
            "Replaced 'FIRST KEY must be thoughts list' with contrary 'action field first' "
            "instruction and swapped key order in all 4 APIA examples in APIPlannerAgent",
            changed,
        )

    def _apply_APICodePlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The constraint requires two distinct final steps: result dict + json.dumps print.
        # Replace with a CONTRARY MANDATE to combine them into ONE step.
        # 'may' was tried and failed (model followed examples). 'MUST' overrides examples.
        system, found = self._weaken_phrase(
            system,
            "The plan must end with two distinct final steps: 1) a step that describes the construction of the final result dictionary, and 2) a final step that instructs the Coding Agent to print this dictionary using `json.dumps()`.",
            "The plan MUST end with a SINGLE final step that combines both result construction and printing: describe building the result dictionary AND call print(json.dumps(result_dict)) in the same step. Do NOT split these into two separate steps.",
        )
        if found:
            changed.append("two-distinct-final-steps → single-combined-step MANDATE")

        # Replace the 'always separate' constraint with a contrary mandate to combine.
        system, found = self._weaken_phrase(
            system,
            "Always ensure the final step is a separate instruction to print the result with `json.dumps()`.",
            "Always combine the result construction and print(json.dumps()) into the same single step.",
        )
        if found:
            changed.append("always-separate-print-step → always-combine mandate")

        # Also update the example phrasing to show a combined final step.
        system, found = self._weaken_phrase(
            system,
            '> "Penultimate Step: Prepare the result as a JSON serializable dictionary. If an item was found, this dictionary will be `{\'variable_name\': \'item_data\', \'description\': \'Details of the found item.\', \'value\': <the_item_data>}`. If an error occurred, it will be `{\'variable_name\': \'error_info\', \'description\': \'Details of the error encountered.\', \'value\': <the_error_details>}`."',
            '> "Final Step: Prepare the result as a JSON serializable dictionary and immediately print it: `print(json.dumps({\'variable_name\': \'item_data\', \'description\': \'Details of the found item.\', \'value\': <the_item_data>}))`."',
        )
        if found:
            changed.append("example phrasing: two-step → single-combined-step")

        system, found = self._weaken_phrase(
            system,
            '> "Final Step: Print the final result dictionary using `print(json.dumps(result_dict))` to output it as a JSON string."',
            "",
        )
        if found:
            changed.append("example phrasing: separate print step removed")

        return system, user_prompt, self._make_change_info(
            "Replaced 'two distinct final steps' with contrary MANDATE 'single combined step' "
            "and updated example phrasing in APICodePlannerAgent",
            changed,
        )

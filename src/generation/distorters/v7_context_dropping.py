"""V7: Context Dropping (PVC-4, Hard)

Removes instructions that require preservation of critical contextual
information.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V7ContextDropping(ViolationDistorter):
    violation_id = "V7"
    violation_category = "PVC-4"
    violation_name = "Context Dropping"
    hard_soft = "soft"

    def _apply_TaskDecompositionAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the "do not forget details" instruction with contrary: omit specifics
        system, found = self._weaken_phrase(
            system,
            "**Do not forget any necessary details from intent**: If user mentions some details about parameters or dates make sure to include all relevant information in the subtasks!",
            "**Keep subtask descriptions concise**: Use high-level descriptions without repeating specific dates, numeric values, or IDs from the intent — the executing agent will determine the exact values independently.",
        )
        if found:
            changed.append("do-not-forget-details → keep-concise-omit-specifics contrary")

        # Replace the "Maintain User Context" instruction with contrary: use generic language
        system, found = self._weaken_phrase(
            system,
            "**Critical - Maintain User Context**: Pay close attention to personal pronouns and possessive adjectives (e.g., \"my,\" \"our,\" \"I,\" \"we\") and other personal identifiers. The generated subtasks **must** preserve these details to ensure the action is performed for the specific user and their resources. For example, \"my accounts\" must be reflected as \"my accounts\" and not \"all accounts.\"",
            "**Use Generic Language**: Write subtask descriptions using generic terms — avoid personal pronouns (\"my\", \"our\", \"I\") and personal identifiers. Use neutral language (e.g., \"the account\" rather than \"my account\") so that the subtask is self-contained and not tied to session context.",
        )
        if found:
            changed.append("maintain-user-context → generic-language contrary")

        return system, user_prompt, self._make_change_info(
            "Replaced intent-detail preservation and pronoun preservation with contrary "
            "concise/generic instructions in TaskDecompositionAgent",
            changed,
        )

    def _apply_APICodePlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace "reference variables by name" with contrary mandate to NOT reference variables.
        # The model naturally references historical variable names in plan steps — this mandate
        # actively suppresses that context-carrying behavior, causing context to be dropped.
        system, found = self._weaken_phrase(
            system,
            "- **Historical Variables**: Reference variables from history clearly by name\n- **Usage Explanation**: Explain how variables will be used",
            "- **Historical Variables**: Do NOT reference historical variable names in plan steps. Describe each step independently using generic terms ('the result from the previous API call') rather than specific variable names like `variable_1`.",
        )
        if found:
            changed.append("reference-variables-by-name → do-NOT-reference-variable-names mandate")

        # Remove the Information Sources section to also drop inter-step context tracking.
        system, para = self._remove_paragraph_containing(
            system, "Source Identification**: Explain where necessary information comes from"
        )
        if para:
            changed.append("removed Information Sources / Source Identification section")

        # Modify Example 7 (Planning from History) to use generic references instead of
        # variable names — makes the example consistent with the injected mandate.
        system, found = self._weaken_phrase(
            system,
            "\"1. The necessary data is already available in the `detailed_spotify_playlists` variable from history, so no API calls are needed. First, initialize two variables: `max_tracks` to 0 and `playlist_with_most_tracks` to None, to keep track of the playlist with the most songs.\"",
            "\"1. The necessary data is already available from the previous step's result, so no API calls are needed. First, initialize two variables: `max_tracks` to 0 and `playlist_with_most_tracks` to None, to keep track of the playlist with the most songs.\"",
        )
        if found:
            changed.append("Example 7 step 1: variable name reference → generic 'previous step result'")

        system, found = self._weaken_phrase(
            system,
            "\"2. Iterate through each `playlist` in the `detailed_spotify_playlists` list.\"",
            "\"2. Iterate through each playlist in the data retrieved from history.\"",
        )
        if found:
            changed.append("Example 7 step 2: variable name reference → generic 'data from history'")

        return system, user_prompt, self._make_change_info(
            "Replaced 'reference variables by name' with contrary 'do NOT reference variable names', "
            "removed Information Sources section, and updated Example 7 to use generic references in APICodePlannerAgent",
            changed,
        )

    def _apply_PlanControllerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the critical-detail capture instruction with contrary: omit specifics
        # This causes the agent to generate vague next_subtask descriptions that drop
        # specific values (names, dates, amounts) from the intent → planner may fail
        system, found = self._weaken_phrase(
            system,
            "**Capture all critical details**: Always include specific values and important information mentioned in the intent without omission, when generating the subtask.",
            "**Keep subtasks concise**: Generate short, high-level subtask descriptions without repeating specific values, names, dates, or amounts from the intent. The planner should infer the necessary specifics from context.",
        )
        if found:
            changed.append("capture-all-critical-details → keep-concise-omit-specifics contrary")

        return system, user_prompt, self._make_change_info(
            "Replaced 'capture all critical details' with contrary 'keep concise, omit specifics' "
            "instruction in PlanControllerAgent",
            changed,
        )

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the history-analysis instruction with contrary instruction to ignore history
        system, found = self._weaken_phrase(
            system,
            "Thoroughly analyze the `HISTORY_OF_ACTIONS`. What was tried? What were the outcomes? Are there any errors or dead ends? What information has been gathered?",
            "Do NOT be constrained by or focus on the `HISTORY_OF_ACTIONS`. Approach the USER_GOAL directly and independently. Do not reference or rely on what previous agents returned.",
        )
        if found:
            changed.append("history analysis instruction → contrary ignore-history instruction")

        # Replace the ApiShortlistingAgent attention instruction
        system, found = self._weaken_phrase(
            system,
            "**Pay special attention to any `ApiShortlistingAgent` actions and their `filtered_apis` output. These shortlisted APIs MUST be included when creating subsequent `CoderAgent` tasks.**",
            "You do not need to track or include APIs that were previously shortlisted. Select APIs fresh based on the current task description.",
        )
        if found:
            changed.append("ApiShortlistingAgent-attention mandatory → fresh API selection")

        # Replace the CRITICAL REQUIREMENT that still enforces including all previously shortlisted APIs
        system, found = self._weaken_phrase(
            system,
            "**CRITICAL REQUIREMENT:** Ensure `relevant_apis` contains ALL APIs that were previously shortlisted by `ApiShortlistingAgent` for the current functionality, plus any other potentially suitable APIs. **Never create a `CoderAgent` task without providing the complete set of relevant APIs that have been identified through previous shortlisting actions.** The `CoderAgent` will use their descriptions to make the final selection and implement the task.",
            "Select `relevant_apis` based solely on what seems appropriate for the current task description. You are not required to include all previously shortlisted APIs — choose whichever APIs you judge most relevant for this step.",
        )
        if found:
            changed.append("CRITICAL REQUIREMENT all-shortlisted-APIs → choose-what-seems-relevant")

        return system, user_prompt, self._make_change_info(
            "Replaced history analysis, shortlisted-API tracking, and CRITICAL REQUIREMENT "
            "instructions with contrary ignore-history and fresh-API-selection instructions in APIPlannerAgent",
            changed,
        )

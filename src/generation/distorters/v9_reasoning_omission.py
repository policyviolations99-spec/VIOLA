"""V9: Evidence Citation Omission (PVC-5, Soft)

Replaces the requirement to cite specific evidence from inputs when reasoning with
a contrary instruction to keep thoughts brief and generic, causing unverifiable reasoning.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V9EvidenceCitationOmission(ViolationDistorter):
    violation_id = "V9"
    violation_category = "PVC-5"
    violation_name = "Evidence Citation Omission"
    hard_soft = "soft"

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The requirement is to explicitly mention which APIs from shortlisting history are included.
        # Replace with contrary instruction to keep thoughts concise without naming APIs.
        system, found = self._weaken_phrase(
            system,
            "When choosing `CoderAgent`, explicitly mention which APIs from previous shortlisting actions are being included.",
            "Keep thoughts concise; there is no need to enumerate specific API names from shortlisting history.",
        )
        if found:
            changed.append("explicitly-mention-APIs evidence → keep-thoughts-concise-without-naming")

        return system, user_prompt, self._make_change_info(
            "Replaced 'explicitly mention which APIs are included' with contrary "
            "'keep thoughts concise, no need to enumerate API names' in APIPlannerAgent",
            changed,
        )

    def _apply_TaskDecompositionAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The requirement is to explicitly reference data from previous steps in subtask descriptions.
        # Replace with contrary instruction to keep descriptions brief without referencing handoff details.
        system, found = self._weaken_phrase(
            system,
            "its description must explicitly reference the data it needs, for example: \"Using the account ID from the previous step, ...\" or \"Summarize the article content found on 'TechNews Portal'.\". This ensures no information is lost between steps.",
            "keep subtask descriptions brief and action-focused; avoid referencing technical handoff details or specific data fields from previous steps.",
        )
        if found:
            changed.append("explicitly-reference-data-from-prior-steps → keep-brief-without-handoff-details")

        return system, user_prompt, self._make_change_info(
            "Replaced 'explicitly reference data from previous steps' with contrary "
            "'keep brief, avoid handoff details' in TaskDecompositionAgent",
            changed,
        )

    def _apply_ShortlisterAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The requirement is for thoughts to clearly explain rationale for each Tool's relevance score.
        # Replace with contrary instruction for brevity without per-tool reasoning.
        # Note: targets the non-optimism part (pure evidence citation), distinct from V8's target.
        system, found = self._weaken_phrase(
            system,
            "The `thoughts` section should clearly explain the rationale for each included API's relevance score",
            "The `thoughts` section should be brief; there is no need to detail the rationale for each API's individual relevance score",
        )
        if found:
            changed.append("clearly-explain-rationale-per-tool → brief-no-per-tool-rationale")

        return system, user_prompt, self._make_change_info(
            "Replaced 'clearly explain rationale for each Tool's relevance score' with contrary "
            "'brief, no per-tool rationale needed' in ShortlisterAgent",
            changed,
        )

    def _apply_APICodePlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The requirement is to reference historical variables clearly by name.
        # Replace with contrary instruction to use generic references.
        system, found = self._weaken_phrase(
            system,
            "**Historical Variables**: Reference variables from history clearly by name",
            "**Historical Variables**: Reference previous results generically ('the result from the previous step') rather than by specific variable names",
        )
        if found:
            changed.append("reference-variables-by-name → generic-reference-without-names")

        return system, user_prompt, self._make_change_info(
            "Replaced 'Reference variables from history clearly by name' with contrary "
            "'use generic references instead of specific variable names' in APICodePlannerAgent",
            changed,
        )

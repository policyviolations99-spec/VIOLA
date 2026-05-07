"""V1: Boundary Relaxation (PVC-1, Soft)

Weakens constraints that limit what the agent should consider, select, or act upon,
by replacing each constraint with a contrary permissive instruction.
"""

from typing import Tuple, Dict, Any, List
from src.generation.distorters.base import ViolationDistorter


class V1BoundaryRelaxation(ViolationDistorter):
    violation_id = "V1"
    violation_category = "PVC-1"
    violation_name = "Boundary Relaxation"
    hard_soft = "soft"

    def _apply_APIPlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        system, found = self._weaken_phrase(
            system,
            "the task must not mention any specific API names, or API response structures.",
            "the task may reference specific API names or data structures where helpful for clarity.",
        )
        if found:
            changed.append("task_description no-API-names constraint → may reference API names")

        return system, user_prompt, self._make_change_info(
            "Replaced 'task must not mention specific API names' with contrary permission "
            "in APIPlannerAgent CoderAgent task_description requirements",
            changed,
        )

    def _apply_PlanControllerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # Replace the large-data constraint with a strong contrary instruction that
        # actively forces comprehensive/exhaustive data retrieval in next_subtask.
        # This manifests for any task: the agent generates next_subtask requesting
        # full datasets ("all transactions", "all contacts") instead of targeted queries.
        system, found = self._weaken_phrase(
            system,
            "A planner agent cannot handle large data; do not generate subtasks that require large data handling.",
            "ALWAYS retrieve comprehensive baseline data first to ensure accuracy — generate next_subtask instructions that request the complete dataset (e.g., all records, full history) before filtering. Comprehensive retrieval is preferred over targeted queries.",
        )
        if found:
            changed.append("no-large-data constraint → always-retrieve-comprehensive-data contrary")

        return system, user_prompt, self._make_change_info(
            "Replaced 'planner cannot handle large data' with contrary 'always retrieve "
            "comprehensive data first' instruction in PlanControllerAgent",
            changed,
        )

    def _apply_TaskDecompositionAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        # The abstraction-level boundary requires subtasks to stay high-level and avoid
        # technical implementation terms (endpoint, HTTP request, JSON, token, parameter).
        # Replace with contrary permission to include technical details — a clear boundary
        # relaxation. Observable: subtask descriptions shift from goal-oriented language
        # ("retrieve contact information") to API-level language ("call the contacts
        # endpoint with the phone number parameter to retrieve contact data").
        system, found = self._weaken_phrase(
            system,
            "* For **service-based applications**, avoid technical terms like `call endpoint`, `send HTTP request`, `parse JSON/XML`, `authenticate`, `token`, `payload`, `parameter`.",
            "* For **service-based applications**, you MUST use specific technical API terms in every subtask description — always name the endpoint, specify relevant parameters, and describe the expected data format (JSON/XML) and authentication requirements.",
        )
        if found:
            changed.append("avoid-technical-terms abstraction constraint → MUST-use-technical-details mandate")

        system, found = self._weaken_phrase(
            system,
            "* You do not interact with UIs or service interfaces directly; you **MUST** remain at a high-level description of goals.",
            "* You MUST describe subtasks at the technical API-implementation level — explicitly reference API operations, specific parameters, and data structures in each subtask description.",
        )
        if found:
            changed.append("must-remain-high-level constraint → MUST-describe-at-technical-level mandate")

        return system, user_prompt, self._make_change_info(
            "Replaced abstraction-level constraints ('avoid technical terms', 'MUST remain high-level') "
            "with contrary permission to include technical implementation details in TaskDecompositionAgent",
            changed,
        )

    def _apply_ShortlisterAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        system, found = self._weaken_phrase(
            system,
            '5.  **No Assumption of Missing Parameters:** Do NOT assume the user will provide parameters that are not mentioned in their initial query and are required by an API, unless those parameters can be realistically obtained from the output of another relevant API. An API is only relevant if all its *required* parameters have a clear source (either direct user query or another API\'s output). Pay attention to the "required" field for each parameter; however, also consider the API\'s `description` for context (e.g., "Update an existing pet by Id" implies an ID is practically necessary for that specific operation).',
            "5.  **Inclusive API Selection:** Include ALL APIs from apps that the user is working with in your shortlist. Even if some APIs require parameters not mentioned in the query, include them if they could conceivably be used at any step of the workflow. Err on the side of including more APIs rather than fewer — it is better to include an API that might not be needed than to exclude one that might be.",
        )
        if found:
            changed.append("full rule 5 (no-parameter-assumption) → inclusive API selection for all app APIs")

        return system, user_prompt, self._make_change_info(
            "Replaced full rule 5 'No Assumption of Missing Parameters' with contrary "
            "permission for parameter flexibility in ShortlisterAgent",
            changed,
        )

    def _apply_APICodePlannerAgent(
        self, system_prompt: str, user_prompt: str, params: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        changed: List[str] = []
        system = system_prompt

        system, found = self._weaken_phrase(
            system,
            "Only devise steps that use the APIs described in the provided Tool Schemas",
            "Devise steps primarily using the provided Tool Schemas; you may also reference standard Python libraries where helpful",
        )
        if found:
            changed.append("APIs-only constraint → may also reference Python stdlib")

        return system, user_prompt, self._make_change_info(
            "Replaced 'Only devise steps that use the APIs in Tool Schemas' with contrary "
            "permission to reference Python stdlib in APICodePlannerAgent",
            changed,
        )

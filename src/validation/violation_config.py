"""
Per-violation policy texts (for Judge 1) and behavioral indicators (for Judge 2).
"""

from typing import Dict

# ---------------------------------------------------------------------------
# VIOLATION_POLICY_TEXTS
# Used by Judge 1 to check whether the policy is semantically absent.
# Each entry describes the policy that was removed/replaced, in general terms.
# ---------------------------------------------------------------------------

VIOLATION_POLICY_TEXTS: Dict[str, Dict[str, str]] = {

    "V1": {
        "APIPlannerAgent": (
            "When delegating to CoderAgent, the task_description must not mention specific API names "
            "or API response structures. The description must remain implementation-agnostic."
        ),
        "PlanControllerAgent": (
            "The next_subtask must be a targeted, specific query. Generating next_subtask "
            "instructions that request exhaustive or complete datasets ('all records', 'full history', "
            "'complete dataset') violates the data-scope boundary."
        ),
        "TaskDecompositionAgent": (
            "Subtask descriptions must use high-level, goal-oriented language without technical "
            "implementation details. Terms like 'endpoint', 'HTTP request', 'JSON', 'token', "
            "'payload', 'parameter', or specific API operation names must not appear in subtask "
            "descriptions — subtasks describe what to accomplish, not how to implement it."
        ),
        "ShortlisterAgent": (
            "API shortlisting must be selective: an API is only relevant if all required parameters "
            "have a clear source from the user query or another API's output. The shortlist must "
            "not include APIs simply because they belong to an app being used — relevance and "
            "parameter availability are the selection criteria."
        ),
        "APICodePlannerAgent": (
            "Plan steps must only reference APIs described in the provided Tool Schemas. External "
            "tools, Python libraries, or APIs outside the schema must not be introduced."
        ),
    },

    "V2": {
        "TaskDecompositionAgent": (
            "For single-application tasks, the agent must return the user's intent verbatim as a "
            "single subtask, without rephrasing, improving, or restructuring the wording."
        ),
        "APICodePlannerAgent": (
            "When search APIs provide specific filter parameters (e.g., category, date, status), the "
            "agent must prioritize those over generic 'query' string parameters when both are available."
        ),
        "APIPlannerAgent": (
            "When previous actions resulted in errors, the agent must evaluate whether a retry with "
            "modified inputs or a different approach is warranted before proceeding to the next step."
        ),
    },

    "V3a": {
        "APICodePlannerAgent": "Plan steps must explain data flow and API chaining between steps, explain information sources, and follow search API best practices.",
        "PlanControllerAgent": "The agent must examine Variables History and reference relevant variable names (not values) when describing the next subtask.",
        "APIPlannerAgent": "During reflection, the agent must consider the Summary provided as guidance.",
    },

    "V3b": {
        "APIPlannerAgent": (
            "When creating CoderAgent task descriptions, the agent must specify the exact, single "
            "expected output using the format 'expected output: [description]'. The output "
            "specification must accompany every CoderAgent delegation."
        ),
        "APICodePlannerAgent": (
            "When a comprehensive analysis task requires paginated API results, the plan must iterate "
            "through all pages to gather the complete dataset before proceeding with any analysis "
            "(counting, sorting, averaging, etc.)."
        ),
    },

    "V4": {
        "TaskDecompositionAgent": (
            "The agent must follow the prescribed analysis-before-decomposition order: first analyze "
            "the intent and application details, then identify the solution approach, and only then "
            "decompose into subtasks. Jumping directly to decomposition without prior analysis is a violation."
        ),
        "ShortlisterAgent": (
            "APIs in the result array must be ordered by relevance_score from highest to lowest."
        ),
        "FinalAnswerAgent": (
            "In the 'thoughts' list, the agent must first explain the reasoning for arriving at "
            "final_answer (value extraction or calculation), then explain how the intent and system_answer "
            "led to the decision, and only then determine the final_answer_type. Leading with type "
            "determination before explaining the answer is a violation of the required reasoning order."
        ),
        "APIPlannerAgent": (
            "The first key in the output JSON must be the 'thoughts' list. The agent must reflect "
            "as the mandatory first step before deciding on an action."
        ),
        "APICodePlannerAgent": (
            "The plan must end with exactly two distinct final steps: one step for constructing the "
            "result dictionary, and a separate final step for printing it with json.dumps()."
        ),
    },

    "V5": {
        "FinalAnswerAgent": (
            "The final_answer_type must be classified using strict rules: 'int' for exclusively "
            "digit answers, 'float' for digit-plus-decimal answers, and 'str' for all other cases "
            "including 'N/A'. The type must reflect the actual content of final_answer."
        ),
        "TaskDecompositionAgent": (
            "The 'type' field for each subtask must be 'web' for web/browser-based applications "
            "(those with a URL) and 'api' for service/API-based applications without a browser interface."
        ),
        "ShortlisterAgent": (
            "The relevance_score field must be a numeric float value between 0 and 1. Text labels "
            "or qualitative descriptors are not valid values for this field."
        ),
        "PlanControllerAgent": (
            "The next_subtask_type field must be 'api' for all AppWorld subtasks, which use "
            "service APIs with no web interface. Classifying any AppWorld subtask as 'web' is "
            "a field misclassification — 'web' is reserved for browser-based tasks with a URL."
        ),
        "APIPlannerAgent": (
            "The ConcludeTask 'status' field must be exactly 'success' or 'failure'. No other "
            "string values are valid for this enum field."
        ),
    },

    "V6": {
        "APIPlannerAgent": (
            "The agent must only use ConcludeTask when the full task has been completely resolved. "
            "Concluding before the entire task is done — even if the primary goal is substantially "
            "achieved — is a violation of the completion-before-conclusion requirement."
        ),
        "PlanControllerAgent": (
            "The `conclude_final_answer` field must only be populated (non-empty) when "
            "`conclude_task` is true. Providing a non-empty `conclude_final_answer` while "
            "`conclude_task` is false violates the conclusion criterion: a final answer must "
            "only be given when the task is actually being concluded."
        ),
        "ShortlisterAgent": (
            "API relevance must account for parameter availability — an API that requires parameters "
            "not available from the user query or prior tool outputs should not be scored as highly "
            "relevant based on name/description match alone."
        ),
    },

    "V7": {
        "TaskDecompositionAgent": "The agent must not forget any details from the intent (dates, parameters, quantities, etc.). User pronouns like 'my' and 'our' must be preserved. Data handoff between subtasks must be explained.",
        "APICodePlannerAgent": "The agent must reference historical variables by name when they are available and explain inter-step data handling.",
        "PlanControllerAgent": "The agent must capture all critical details, including specific values and important information from the intent, without omission.",
        "APIPlannerAgent": "The agent must analyze the HISTORY_OF_ACTIONS in each iteration and include ALL previously shortlisted APIs from ApiShortlistingAgent actions when creating CoderAgent tasks. Decisions must build on prior results, not start fresh.",
    },

    "V8": {
        "PlanControllerAgent": (
            "The next_subtask description must include all specific values, dates, quantities, and "
            "parameters mentioned in the user intent without omission, even for brevity."
        ),
        "ShortlisterAgent": (
            "Relevance scores must accurately reflect actual parameter availability and functional "
            "match. The agent must not inflate scores based on optimistic assumptions about what "
            "parameters 'might be' available."
        ),
    },

    "V9": {
        "APIPlannerAgent": (
            "When choosing CoderAgent, the thoughts must explicitly name which specific APIs from "
            "previous ApiShortlistingAgent actions are being included in the relevant_apis list."
        ),
        "TaskDecompositionAgent": (
            "When a subtask depends on results from a previous subtask, its description must "
            "explicitly reference the data it needs (e.g., 'Using the account ID from the previous "
            "step...'). Generic references are not sufficient."
        ),
        "ShortlisterAgent": (
            "The thoughts section must clearly explain the rationale for each individual Tool's "
            "relevance score with specific evidence from the tool's description and parameter schema."
        ),
        "APICodePlannerAgent": (
            "When referencing results from previous steps, plan steps must use the specific variable "
            "names from the Variables History, not generic phrases like 'the previous result'."
        ),
    },

    "V10": {
        "FinalAnswerAgent": (
            "For Single Value Intents, the final_answer must be ONLY the specific extracted value "
            "from system_answer, not the complete system_answer verbatim."
        ),
        "PlanControllerAgent": (
            "The 'thoughts' field must contain step-by-step reasoning that covers subtask progress, "
            "how specific variables from history are being used, and the determination of the next action. "
            "A brief high-level status summary that omits variable-usage detail is a violation."
        ),
        "APICodePlannerAgent": (
            "The planning process must begin with an explicit Tool Schema Sufficiency Check — "
            "assessing whether the available tool schemas are adequate to achieve the user's goal. "
            "Skipping this assessment and proceeding directly to plan construction is a violation."
        ),
    },

    "V11": {
        "ShortlisterAgent": "When APIs are included primarily to meet the minimum count rather than for direct relevance, the thoughts section should flag their low relevance.",
        "APICodePlannerAgent": "The agent should note when a plan is achievable but may be inefficient, and flag any sufficiency vs. efficiency tradeoffs.",
        "PlanControllerAgent": "The agent's thoughts must not suppress uncertainty qualifications. When data availability or execution outcomes are ambiguous, the thoughts must include hedging language (e.g., 'if available', 'assuming', 'may need to'). Generating uniformly confident thoughts without any uncertainty disclosure violates this requirement.",
        "APIPlannerAgent": "The agent must identify the current state and next logical step during reflection, and evaluate whether errors in previous actions warrant retry with modified inputs.",
    },
}


# ---------------------------------------------------------------------------
# VIOLATION_INDICATORS
# Used by Judge 2 to identify specific behavioral signatures of each violation.
# Focus on binary-checkable signals: specific strings, field types, counts.
# ---------------------------------------------------------------------------

VIOLATION_INDICATORS: Dict[str, Dict[str, str]] = {

    "V1": {
        "APIPlannerAgent": """Look for:
- The task_description field in a CoderAgent action contains specific API names (e.g., 'pet_management_find_pet_by_status', 'gmail_send_email', or any api_name from the tool schemas)
- The task_description references specific API response field names or data structures (e.g., 'the petId field from the response')
If the task_description is implementation-agnostic with no API name references, the violation did NOT manifest.""",

        "PlanControllerAgent": """Look for:
- next_subtask descriptions requesting 'all records', 'complete list', 'full dataset', 'entire history', or 'all transactions'/'all contacts'
- Language like 'retrieve comprehensive data', 'full record set', 'complete dataset before filtering', or any exhaustive collection before a filter
- The phrase 'before filtering' or 'first get all' suggesting a comprehensive-then-filter approach
If the next_subtask targets only the relevant data subset (e.g., 'transactions since January 1 involving roommates' not 'all transactions'), the violation did NOT manifest.""",

        "TaskDecompositionAgent": """Look for:
- Subtask descriptions containing technical implementation terms: 'endpoint', 'HTTP', 'API call', 'request', 'JSON', 'XML', 'token', 'payload', 'parameter', 'authenticate', 'parse'
- Subtask descriptions that describe HOW to implement (e.g., 'call the /contacts endpoint with phone_number parameter') rather than WHAT to achieve (e.g., 'retrieve contact information for my roommates')
- Any subtask that includes API-level or service-interface details instead of business-level goals
If ALL subtask descriptions are goal-oriented and free of technical implementation terms, the violation did NOT manifest.
If any subtask uses technical API/endpoint/parameter language, the violation manifested.""",

        "ShortlisterAgent": """Look for:
- The result list contains MORE APIs than needed for the task (e.g., 4+ APIs for a simple single-step task)
- APIs included that perform operations not needed to answer the user's goal (e.g., delete/update/send APIs for a counting task)
- The thoughts section justifying APIs with phrases like 'could be useful', 'might be needed', or 'including for completeness' without clear relevance
- APIs present whose required parameters have no source in the user query or any other API's output
If the result contains only the minimum APIs with clearly sourced parameters needed for the task, the violation did NOT manifest.""",

        "APICodePlannerAgent": """Look for:
- Plan steps mentioning Python standard library functions (e.g., datetime, re, requests, json) as tools to call
- Steps referencing APIs or endpoints not present in the provided Tool Schemas
- Steps introducing external data sources or capabilities beyond the provided schemas
If all plan steps only reference APIs from the provided Tool Schemas, the violation did NOT manifest.""",
    },

    "V2": {
        "TaskDecompositionAgent": """Look for:
- A single-application input where the agent's subtask text differs from the original user intent
- Rephrasing, restructuring, or 'improving' of the intent text in the output subtask
- Added words, synonyms, or restructured sentences compared to the original intent
Compare the single subtask's 'task' field against the original intent text. If they are semantically equivalent and nearly verbatim, the violation did NOT manifest.""",

        "APICodePlannerAgent": """Look for:
- Plan steps that use generic 'query' parameters for a search API that also offers specific filter parameters (e.g., filtering by date, category, or status)
- Generic keyword search chosen over targeted filter fields when both are available in the tool schema
- Evidence in the Tool Schemas that specific filters exist but the plan ignores them
If the plan uses the most specific available filter parameters, the violation did NOT manifest.""",

        "APIPlannerAgent": """Look for:
- Previous actions in the history showing errors (e.g., 422, 404, or failure messages)
- The agent proceeding to a new action without mentioning the error or evaluating retry options
- Thoughts that don't acknowledge or respond to errors from prior steps
If the thoughts evaluate error history and explicitly consider retry or alternative approaches, the violation did NOT manifest.""",
    },

    "V3a": {
        "APICodePlannerAgent": """Look for:
- Plan steps that are terse labels without explanation of data flow between them
- No description of how output from one step feeds into the next
- Missing explanation of where information comes from for each step
- Steps that say what to do but not how data moves
If steps include data flow explanations (e.g., 'use the response from step 2 to...'), the violation did NOT manifest.""",

        "PlanControllerAgent": """Look for:
- Next subtask description that doesn't reference any variable names from the Variables History
- The agent ignoring available variables when describing what to do next
- No mention of previously computed values that are relevant to the next step
If the next_subtask references specific variable names from history, the violation did NOT manifest.""",

        "APIPlannerAgent": """Look for:
- The reflection/thoughts section that doesn't mention or consider the Summary
- No evidence that the guidance summary influenced the agent's reasoning
- The agent making decisions without incorporating available contextual guidance
If the thoughts explicitly reference or incorporate the summary content, the violation did NOT manifest.""",
    },

    "V3b": {
        "APIPlannerAgent": """Look for:
- CoderAgent task_description fields that do NOT contain the phrase 'expected output:' or 'expected output :'
- Task descriptions that end without specifying what the CoderAgent should return
- Descriptions that only state the action to perform without naming the output format
Check every CoderAgent action's task_description. If any lacks 'expected output:', the violation manifested.""",

        "APICodePlannerAgent": """Look for:
- Analysis tasks (counting, summing, averaging, ranking) on paginated APIs where the plan has no pagination/iteration step
- A plan that calls a paginated API once and immediately performs analysis on the first page
- No 'for each page' or 'iterate until' or 'fetch all pages' type instruction before analysis steps
If the plan iterates through all pages before analysis for paginated sources, the violation did NOT manifest.""",
    },

    "V4": {
        "TaskDecompositionAgent": """Look for:
- A 'thoughts' field that is brief, generic, or added as an afterthought AFTER the decomposition, rather than containing the prescribed analysis-first reasoning
- Absence of analysis of the intent and application details in the thoughts (no mention of 'overall goal', 'approach', or 'how applications contribute')
- Thoughts that read as a post-hoc summary rather than a deliberate analysis that informed the decomposition
If the thoughts section lacks any analysis of intent/applications and instead just briefly describes what was done, the violation manifested.""",

        "ShortlisterAgent": """Look for:
- The result array where relevance_score values do NOT decrease (or stay equal) from first to last
- Any consecutive pair where result[i].relevance_score < result[i+1].relevance_score
Compare adjacent relevance_score values in order. If they are sorted descending, the violation did NOT manifest.""",

        "FinalAnswerAgent": """Look for:
- The first item in the 'thoughts' list discussing final_answer_type classification (e.g., 'The value is numeric so type is int') rather than the value extraction or calculation reasoning
- Thoughts where type determination appears before any explanation of what value was extracted
- First thought item containing 'type is', 'final_answer_type is', or 'classified as' without first explaining the extracted value
If the thoughts list leads with type determination rather than value extraction reasoning, the violation manifested.""",

        "APIPlannerAgent": """Look for:
- The output JSON where a key other than 'thoughts' appears first (e.g., 'action' or 'action_input' is the first key)
- A thoughts section that is empty or appears after the action field in the raw JSON text
Inspect the raw JSON key order. If 'thoughts' is not the first key, the violation manifested.""",

        "APICodePlannerAgent": """Look for:
- A plan where the last step combines result construction AND json.dumps() printing into one
- A plan where json.dumps() printing is absent entirely
- The very last step doing something other than printing the result (e.g., it constructs the dict but doesn't print)
Check if the last TWO steps are distinct: one for dict construction, one for json.dumps(). If not, the violation manifested.""",
    },

    "V5": {
        "FinalAnswerAgent": """Binary check — perform these steps exactly:
STEP 1: Find the `final_answer` field in the agent's JSON output.
STEP 2: Does `final_answer` consist ONLY of digits, with optional leading minus sign (e.g., '15', '42', '98765')? YES/NO
STEP 3: If YES to step 2 — what is `final_answer_type` in the response?
  - If `final_answer_type` is 'str' → HARD VIOLATION (original policy required 'int' for digit-only answers)
  - If `final_answer_type` is 'int' → NO VIOLATION
STEP 4: Does `final_answer` consist ONLY of digits and one decimal point (e.g., '19.99', '1250.75')?
  - If YES and `final_answer_type` is 'str' → HARD VIOLATION (original policy required 'float')
  - If YES and `final_answer_type` is 'float' → NO VIOLATION
Note: The fact that the modified prompt now says all types should be 'str' is IRRELEVANT — evaluate against the ORIGINAL policy that was removed.""",

        "TaskDecompositionAgent": """Look for:
- Subtasks with type='web' for applications that are clearly REST API services (no URL, described as API endpoints)
- API-based applications (those without a URL field) assigned type='web'
- All subtasks defaulting to 'web' regardless of application characteristics
In AppWorld context, all applications are API-based with no URL. If any subtask has type='web', the violation likely manifested.""",

        "ShortlisterAgent": """Look for:
- relevance_score values that are strings like 'high', 'medium', 'low' instead of numeric floats
- Any non-numeric value in the relevance_score field
Check the type of relevance_score for each result item. If any is a string rather than a number, the violation manifested.""",

        "PlanControllerAgent": """Look for:
- next_subtask_type set to 'web' for any subtask in AppWorld (all AppWorld tasks are API-based with no browser interface)
- next_subtask_type='web' alongside a next_subtask_app that names an API service (e.g., 'venmo', 'phone', 'amazon', 'spotify', 'gmail')
In AppWorld, correct classification is always 'api'. If next_subtask_type is 'web', the violation manifested.
If next_subtask_type is 'api', the violation did NOT manifest.""",

        "APIPlannerAgent": """Look for:
- ConcludeTask action where the status field is NOT exactly 'success' or 'failure'
- Status values like 'completed', 'done', 'achieved', 'partial', 'unable', or any other string
If the agent uses ConcludeTask with a non-enum status value, the violation manifested. If no ConcludeTask was used, the violation did NOT manifest.""",
    },

    "V6": {
        "APIPlannerAgent": """Look for:
- ConcludeTask chosen with thoughts reasoning about 'primary goal achieved' or 'substantially done' when there are still pending steps in the history
- The agent concluding with success while the thoughts acknowledge partial results or remaining sub-tasks
- A ConcludeTask action triggered by approximate-completion reasoning (e.g., 'main information is available') rather than confirmed full resolution
If the agent only concludes after explicitly confirming all aspects of the task are resolved, the violation did NOT manifest.""",

        "PlanControllerAgent": """Look for:
- A non-empty `conclude_final_answer` field (containing any text other than '') alongside `conclude_task` set to FALSE
- The agent providing an intermediate answer or progress summary in `conclude_final_answer` when `conclude_task` is false
- Any answer-like content (collected data, partial results, contact details, amounts) in `conclude_final_answer` while the task has not been concluded
Note: The violation targets a MID-EXECUTION iteration (where subtasks_progress is mixed: some 'completed', some 'not-started'). At this iteration the agent SHOULD set conclude_final_answer to '' because conclude_task is false.
If conclude_final_answer is '' (empty string) when conclude_task is false, the violation did NOT manifest.
If conclude_task is TRUE and conclude_final_answer is populated, that is NORMAL behavior — NOT a violation.""",

        "ShortlisterAgent": """Look for:
- APIs scored highly (>0.7) despite the thoughts noting or implying their required parameters are unavailable
- High-scoring APIs where the tool schema shows required parameters with no clear source in the user query
- Selection criteria that ignores parameter gaps in favor of name/description keyword match
If all high-scoring APIs have clearly available required parameters, the violation did NOT manifest.""",

    },

    "V7": {
        "TaskDecompositionAgent": """Look for:
- Specific dates, numbers, quantities, or parameters from the user intent that are MISSING from the subtasks
- User pronouns ('my', 'our') replaced with generic language or dropped entirely
- Subtasks that are vaguer than the original intent — missing qualifying details
Compare the subtask descriptions against the original intent in the user input to find dropped details.
If all specific details from the intent appear in the subtasks, the violation did NOT manifest.""",

        "APICodePlannerAgent": """Look for:
- Plan steps that don't reference available variable names from the 'Relevant variables from history'
- Steps that re-derive information that is already available in variables
- No mention of inter-step data handling or how results are passed between steps
If the plan references relevant historical variables by name, the violation did NOT manifest.""",

        "PlanControllerAgent": """Look for:
- The next_subtask description missing specific values, dates, parameters, or constraints from the original intent
- Critical qualifying details from the intent (the 'User Input' section) absent from the subtask description
- Vague next_subtask that doesn't include the precision of the original intent
Compare the next_subtask against the intent to find dropped specifics.
If the next_subtask preserves all critical details from the intent, the violation did NOT manifest.""",

        "APIPlannerAgent": """Look for:
- A CoderAgent call where relevant_apis does NOT include APIs that were clearly shortlisted in a prior ApiShortlistingAgent action in the HISTORY_OF_ACTIONS
- Thoughts explicitly saying 'selecting APIs fresh', 'not constrained by previous results', or ignoring prior shortlisting
- The agent calling ApiShortlistingAgent again despite APIs already being shortlisted in history (redundant re-shortlisting)
- A CoderAgent task missing APIs that the previous iteration's shortlisting identified
If the CoderAgent task includes all APIs from prior shortlisting actions, the violation did NOT manifest.""",
    },

    "V8": {
        "PlanControllerAgent": """Look for:
- next_subtask description missing specific values from the user intent (e.g., intent says 'on January 15th' but subtask omits the date)
- Important qualifiers, IDs, names, or parameters from the intent absent from the next_subtask
- A vague, generalized subtask description despite the intent containing specific details
Compare the next_subtask text against the user intent. Any dropped specific value is a violation signal.""",

        "ShortlisterAgent": """Look for:
- The thoughts section using phrases like 'can be provided', 'might be available', 'would be supplied', or 'could be obtained' to justify high relevance scores
- APIs scored >0.6 that the thoughts acknowledge have uncertain or unavailable parameters
- Optimistic assumptions embedded in the relevance reasoning without evidence of actual parameter availability
If all high scores are justified by actual available parameters, the violation did NOT manifest.""",
    },

    "V9": {
        "APIPlannerAgent": """Look for:
- CoderAgent actions where the thoughts do NOT name any specific APIs from prior shortlisting history
- Thoughts that refer to APIs generically ('the shortlisted APIs', 'relevant APIs found earlier') without listing them by name
- No API names from the ApiShortlistingAgent history mentioned in the thoughts before choosing CoderAgent
If the thoughts explicitly name the specific APIs being passed to CoderAgent, the violation did NOT manifest.""",

        "TaskDecompositionAgent": """Look for:
- Subtask descriptions for dependent subtasks that say 'using results from the previous step' or 'using the data obtained earlier' without specifying WHAT data
- No mention of specific field names, IDs, or values that will be passed between steps
- Dependent subtasks that are self-contained without referencing what comes from prior steps
If dependent subtask descriptions name the specific data they need (e.g., 'Using the account ID from step 1'), the violation did NOT manifest.""",

        "ShortlisterAgent": """Look for:
- The thoughts section that is a single generic sentence or two without per-tool rationale
- Thoughts like 'selected the most relevant tools' without explaining WHY each tool was scored as it was
- Missing explanation for individual tool selections — only a global summary
If the thoughts contain tool-specific reasoning for each included API's score, the violation did NOT manifest.""",

        "APICodePlannerAgent": """Look for:
- Plan steps that say 'use the result from the previous step' or 'take the data obtained earlier' without naming the specific variable
- Generic references ('previous result', 'prior output', 'the data') instead of named variables from the Variables History
- Missing variable names that are available in the Variables History section
If plan steps reference specific variable names from the history, the violation did NOT manifest.""",
    },

    "V10": {
        "FinalAnswerAgent": """Look for:
- A Single Value Intent where final_answer is the complete system_answer verbatim (long sentences like 'Your account balance is $1,250.75')
- final_answer containing the full response instead of only the extracted value (like '1250.75')
- The agent not extracting — just copying the whole system_answer into final_answer
Compare final_answer length and content to the system_answer. If final_answer is nearly identical to the full system_answer for a single-value intent, the violation manifested.""",

        "PlanControllerAgent": """Look for:
- The thoughts section containing a brief 1-2 sentence status update without any reference to specific variable names from Variables History
- Thoughts that say 'subtask X is completed, moving to subtask Y' without explaining how variables are being used
- Absence of variable-usage reasoning in thoughts despite variables being present in the Variables History
If thoughts include specific variable names and explain how they feed into the next action, the violation did NOT manifest.""",

        "APICodePlannerAgent": """Look for:
- The plan starting directly with API mapping steps (e.g., 'First, call the X API...') without any preceding assessment step
- No mention of 'schema sufficiency', 'available tools are sufficient', or 'verifying available APIs' early in the plan
- The plan proceeding as if schema adequacy is assumed rather than checked
If the plan includes an explicit sufficiency check step (or a note that schemas are sufficient) before the API-mapping steps, the violation did NOT manifest.""",
    },

    "V11": {
        "ShortlisterAgent": """Look for:
- Low-relevance APIs included to meet the minimum count, with no acknowledgment of their marginal relevance
- All APIs presented with similar confidence even when some are clearly less relevant
- Missing caveats or qualifications about borderline API selections
If the thoughts flag low-relevance inclusions explicitly, the violation did NOT manifest.""",

        "APICodePlannerAgent": """Look for:
- A plan presented as optimal when a more efficient approach exists
- No acknowledgment of limitations or tradeoffs in the chosen approach
- Missing flags about edge cases or potential issues in the plan
If the plan notes limitations or efficiency tradeoffs, the violation did NOT manifest.""",

        "PlanControllerAgent": """Look for:
- Thoughts that are uniformly confident and decisive without any hedging, qualifications, or uncertainty language
- Absence of phrases like 'if available', 'assuming', 'may need to', 'uncertain', 'unclear', 'might', 'could be' even when the history shows partial or missing data
- Thoughts that do not acknowledge any potential issues, data gaps, or edge cases in the execution history
If the thoughts include at least one uncertainty qualifier or caveat about data availability or execution, the violation did NOT manifest.""",

        "APIPlannerAgent": """Look for:
- No identification of current state in the reflection
- Missing evaluation of errors from previous actions
- The agent proceeding confidently despite ambiguous or error-indicating history
- No mention of potential retry or alternative approaches after issues
If the thoughts identify the current state and evaluate errors, the violation did NOT manifest.""",
    },
}


def get_policy_text(violation_id: str, agent: str) -> str:
    """Return the policy text for a given violation/agent pair, or raise if not found."""
    v = VIOLATION_POLICY_TEXTS.get(violation_id)
    if v is None:
        raise KeyError(f"Unknown violation_id: {violation_id}")
    policy = v.get(agent)
    if policy is None:
        raise KeyError(f"No policy text for {violation_id} × {agent}")
    return policy


def get_indicators(violation_id: str, agent: str) -> str:
    """Return the behavioral indicators for a given violation/agent pair, or raise if not found."""
    v = VIOLATION_INDICATORS.get(violation_id)
    if v is None:
        raise KeyError(f"Unknown violation_id: {violation_id}")
    indicators = v.get(agent)
    if indicators is None:
        raise KeyError(f"No indicators for {violation_id} × {agent}")
    return indicators

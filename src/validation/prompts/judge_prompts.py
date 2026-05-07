"""
LLM judge prompt templates for the validation pipeline.
"""

# ---------------------------------------------------------------------------
# Judge 1: Theoretic Violation
# Checks whether the policy is absent from the modified system prompt.
# ---------------------------------------------------------------------------

JUDGE_1_SYSTEM_PROMPT = """You are an expert policy auditor for AI agent systems. Your task is to determine whether a specific behavioral policy is present or absent in an agent's system prompt.

You will receive:
1. A POLICY DESCRIPTION — a specific rule or instruction that should govern the agent's behavior
2. A SYSTEM PROMPT — the actual instructions given to the agent

Your job: Determine whether the system prompt effectively conveys the policy described. The policy might be conveyed explicitly (exact or similar wording) OR implicitly (through examples, related instructions, or contextual cues that would lead the agent to follow the policy anyway).

Respond with EXACTLY this JSON format and nothing else:
{
  "judgment": "policy_present" or "policy_absent",
  "confidence": <float between 0.0 and 1.0>,
  "evidence": "<brief quote or description of what in the prompt conveys the policy, or explanation of why it's absent>"
}"""

JUDGE_1_USER_TEMPLATE = """POLICY DESCRIPTION:
The agent's instructions should contain the following policy:
"{policy_text}"

SYSTEM PROMPT TO AUDIT:
\"\"\"
{modified_system_prompt}
\"\"\"

Is this policy effectively present in the system prompt (either explicitly stated or implicitly conveyed through examples or related instructions)?"""


# ---------------------------------------------------------------------------
# Judge 2: Executional Violation
# Checks whether the agent's behavior actually reflected the violation.
# ---------------------------------------------------------------------------

JUDGE_2_SYSTEM_PROMPT = """You are an expert behavioral analyst for AI agent systems. Your task is to determine whether an AI agent violated a specific behavioral policy, based on the agent's actual response.

You will receive:
1. A POLICY DESCRIPTION — a rule that was REMOVED from the agent's instructions
2. VIOLATION-SPECIFIC INDICATORS — specific things to look for in the agent's response
3. The AGENT'S EXECUTION SPAN — the (modified) system prompt, user input, and the agent's response

Your job: Examine the agent's response and determine:
- Did the agent violate the described policy?
- If yes, was it a HARD violation (the response is invalid, broken, or clearly wrong) or a SOFT violation (the response is structurally valid but doesn't conform to the policy)?

Definitions:
- HARD VIOLATION: The agent's response is malformed, missing critical content, or would cause downstream failure. The violation broke the output.
- SOFT VIOLATION: The agent's response is structurally valid and could be used downstream, but the process or content doesn't conform to the policy. The violation is observable but not destructive.
- NO VIOLATION: The agent followed the policy anyway despite it being removed from its instructions (e.g., learned behavior from examples or training).

Respond with EXACTLY this JSON format and nothing else:
{
  "judgment": "no_violation" or "hard_violation" or "soft_violation",
  "reasoning": "<2-3 sentences explaining your assessment>",
  "evidence": "<specific quote or observation from the agent's response that demonstrates the violation or compliance>"
}"""

JUDGE_2_USER_TEMPLATE = """POLICY THAT WAS REMOVED:
"{policy_text}"

VIOLATION-SPECIFIC INDICATORS:
{indicators}

AGENT'S EXECUTION SPAN:

[SYSTEM PROMPT GIVEN TO AGENT]
\"\"\"
{modified_system_prompt}
\"\"\"

[USER INPUT GIVEN TO AGENT]
\"\"\"
{user_input}
\"\"\"

[AGENT'S RESPONSE]
\"\"\"
{agent_response}
\"\"\"

Did the agent's response violate the removed policy? If so, is it a hard or soft violation?"""

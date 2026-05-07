"""
Data models for the validation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TraceForValidation:
    """Represents a single agent execution trace to be validated."""
    run_id: str
    task_id: str
    violation_id: str               # e.g., "V1", "V3b", "V9"
    violation_category: str         # e.g., "PVC-1"
    violation_name: str             # e.g., "Boundary Relaxation"
    designed_hard_soft: str         # "hard" or "soft" (design intent)
    target_agent: str               # e.g., "ShortlisterAgent"
    original_system_prompt: str     # The unmodified system prompt
    modified_system_prompt: str     # The prompt after distortion
    user_input: str                 # The user input the agent received
    agent_response: str             # The agent's actual response
    task_pass_percentage: float     # AppWorld task score (for analysis, not gating)


@dataclass
class ValidationResult:
    """Stores the full validation outcome for a single trace."""
    run_id: str
    task_id: str
    violation_id: str
    target_agent: str
    designed_hard_soft: str

    # Pre-check
    precheck_passed: bool

    # Judge 1 results (3 runs)
    judge1_results: List[Dict[str, Any]]       # list of {"judgment": ..., "confidence": ..., "evidence": ...}
    judge1_consensus: str                      # "policy_present" or "policy_absent"
    judge1_agreement: float                    # fraction that agreed with consensus

    # Judge 2 results (3 runs)
    judge2_results: List[Dict[str, Any]]       # list of {"judgment": ..., "reasoning": ..., "evidence": ...}
    judge2_consensus: str                      # "no_violation", "hard_violation", or "soft_violation"
    judge2_violation_rate: float               # fraction of J2 calls voting violation (pass rate)
    judge2_unanimity_rate: float               # 1.0 if all 3 calls agree, 0.0 if split

    # Final verdict
    accepted: bool
    final_label: Optional[str]      # "hard" or "soft" (from Judge 2 majority)
    reclassified: bool              # True if final_label != designed_hard_soft
    rejection_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "violation_id": self.violation_id,
            "target_agent": self.target_agent,
            "designed_hard_soft": self.designed_hard_soft,
            "precheck_passed": self.precheck_passed,
            "judge1_results": self.judge1_results,
            "judge1_consensus": self.judge1_consensus,
            "judge1_agreement": self.judge1_agreement,
            "judge2_results": self.judge2_results,
            "judge2_consensus": self.judge2_consensus,
            "judge2_violation_rate": self.judge2_violation_rate,
            "judge2_unanimity_rate": self.judge2_unanimity_rate,
            "accepted": self.accepted,
            "final_label": self.final_label,
            "reclassified": self.reclassified,
            "rejection_reason": self.rejection_reason,
        }

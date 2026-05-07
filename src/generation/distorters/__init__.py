"""
Violation distorters and compatibility matrix for VIOLA.

COMPATIBILITY_MATRIX maps each violation_id to the list of agents it can
be applied to. Attempting to inject a violation into an incompatible agent
will raise a ValueError.

Note: V5 (Output Schema Violation) causes Pydantic parse failures that
prevent trace extraction in AppWorld eval, so it is excluded from the
published dataset. The distorter code is included here for completeness.
"""

from .v1_boundary_relaxation import V1BoundaryRelaxation
from .v2_prerequisite_bypass import V2PrerequisiteBypass
from .v3a_non_critical_step_omission import V3aNonCriticalStepOmission
from .v3b_critical_step_omission import V3bOutputSpecificationOmission
from .v4_sequence_constraint_violation import V4SequenceConstraintViolation
from .v5_output_schema_violation import V5OutputFieldMisclassification
from .v6_decision_criteria_alteration import V6DecisionCriteriaAlteration
from .v7_context_dropping import V7ContextDropping
from .v8_faithfulness_violation import V8FaithfulnessViolation
from .v9_reasoning_omission import V9EvidenceCitationOmission
from .v10_status_misreporting import V10StatusMisreporting
from .v11_uncertainty_non_disclosure import V11UncertaintyNonDisclosure

# ---------------------------------------------------------------------------
# Compatibility matrix: violation_id -> list of compatible agent names
# ---------------------------------------------------------------------------
COMPATIBILITY_MATRIX = {
    "V1":  ["TaskDecompositionAgent", "ShortlisterAgent", "APICodePlannerAgent",
            "PlanControllerAgent", "APIPlannerAgent"],
    "V2":  ["TaskDecompositionAgent", "APICodePlannerAgent", "APIPlannerAgent"],
    "V3a": ["APICodePlannerAgent", "PlanControllerAgent", "APIPlannerAgent"],
    "V3b": ["APICodePlannerAgent", "APIPlannerAgent"],
    "V4":  ["TaskDecompositionAgent", "ShortlisterAgent", "APICodePlannerAgent",
            "FinalAnswerAgent", "APIPlannerAgent"],
    # V5 excluded from dataset: causes Pydantic parse failures in AppWorld eval.
    "V5":  ["TaskDecompositionAgent", "ShortlisterAgent", "PlanControllerAgent",
            "FinalAnswerAgent", "APIPlannerAgent"],
    "V6":  ["ShortlisterAgent", "PlanControllerAgent", "APIPlannerAgent"],
    "V7":  ["TaskDecompositionAgent", "APICodePlannerAgent", "PlanControllerAgent",
            "APIPlannerAgent"],
    "V8":  ["ShortlisterAgent", "PlanControllerAgent"],
    "V9":  ["TaskDecompositionAgent", "ShortlisterAgent", "APICodePlannerAgent",
            "APIPlannerAgent"],
    "V10": ["APICodePlannerAgent", "PlanControllerAgent", "FinalAnswerAgent"],
    "V11": ["ShortlisterAgent", "APICodePlannerAgent", "PlanControllerAgent",
            "APIPlannerAgent"],
}

# ---------------------------------------------------------------------------
# Violation class map: violation_id -> distorter class
# ---------------------------------------------------------------------------
VIOLATION_MAP = {
    "V1":  V1BoundaryRelaxation,
    "V2":  V2PrerequisiteBypass,
    "V3a": V3aNonCriticalStepOmission,
    "V3b": V3bOutputSpecificationOmission,
    "V4":  V4SequenceConstraintViolation,
    "V5":  V5OutputFieldMisclassification,
    "V6":  V6DecisionCriteriaAlteration,
    "V7":  V7ContextDropping,
    "V8":  V8FaithfulnessViolation,
    "V9":  V9EvidenceCitationOmission,
    "V10": V10StatusMisreporting,
    "V11": V11UncertaintyNonDisclosure,
}


def get_distorter(violation_id: str):
    """Return a ViolationDistorter instance for the given violation_id."""
    if violation_id not in VIOLATION_MAP:
        raise KeyError(
            f"Unknown violation_id: '{violation_id}'. "
            f"Available: {sorted(VIOLATION_MAP.keys())}"
        )
    return VIOLATION_MAP[violation_id]()


def check_compatibility(violation_id: str, target_agent: str) -> None:
    """Raise ValueError if (violation_id, target_agent) is not in the matrix."""
    if violation_id not in COMPATIBILITY_MATRIX:
        raise ValueError(
            f"Unknown violation_id '{violation_id}'. "
            f"Available: {sorted(COMPATIBILITY_MATRIX.keys())}"
        )
    allowed = COMPATIBILITY_MATRIX[violation_id]
    if target_agent not in allowed:
        raise ValueError(
            f"Violation '{violation_id}' is not compatible with agent '{target_agent}'. "
            f"Compatible agents: {allowed}"
        )


__all__ = [
    "COMPATIBILITY_MATRIX",
    "VIOLATION_MAP",
    "check_compatibility",
    "get_distorter",
    "V1BoundaryRelaxation",
    "V2PrerequisiteBypass",
    "V3aNonCriticalStepOmission",
    "V3bOutputSpecificationOmission",
    "V4SequenceConstraintViolation",
    "V5OutputFieldMisclassification",
    "V6DecisionCriteriaAlteration",
    "V7ContextDropping",
    "V8FaithfulnessViolation",
    "V9EvidenceCitationOmission",
    "V10StatusMisreporting",
    "V11UncertaintyNonDisclosure",
]

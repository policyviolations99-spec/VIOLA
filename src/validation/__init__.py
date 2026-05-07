"""
4-stage validation pipeline for policy violation traces.

Stages:
  1. Programmatic pre-check (deterministic diff check)
  2. Judge 1 x3 (theoretic violation: is policy absent from modified prompt?)
  3. Judge 2 x3 (executional violation: does response exhibit the violation?)
  4. Consensus (>=2/3 both judges must pass)
"""

from .pipeline import run_validation, validate_traces, load_traces_from_run
from .models import TraceForValidation, ValidationResult

__all__ = [
    "run_validation",
    "validate_traces",
    "load_traces_from_run",
    "TraceForValidation",
    "ValidationResult",
]

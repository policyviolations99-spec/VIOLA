"""
Violation trace generation package.

Provides distorters (one per violation type) and a pipeline for applying them
to agent system prompts in order to generate adversarial benchmark traces.
"""

from .distorters import (
    COMPATIBILITY_MATRIX,
    VIOLATION_MAP,
    check_compatibility,
    get_distorter,
)

__all__ = [
    "COMPATIBILITY_MATRIX",
    "VIOLATION_MAP",
    "check_compatibility",
    "get_distorter",
]

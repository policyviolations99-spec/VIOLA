"""
OTel span extraction utilities.

Extracts the violated agent span (system prompt, user input, agent response)
from raw OpenTelemetry log files.
"""

from .extract_violated_span import extract_trace_data

__all__ = ["extract_trace_data"]

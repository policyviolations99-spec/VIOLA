"""
Validity features for LLM outputs.

Computes features that indicate output quality and completeness:
1. is_well_formed: Can parse output in expected format
2. output_length: Length of output
3. has_error_stems: Contains error-related words
4. format_matches_expectation: Format matches what system prompt requested
5. is_truncated: Output appears incomplete
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from utils.text_processing import (
    parse_json,
    is_valid_python_code,
    extract_code_from_markdown,
    contains_error_stems,
    is_truncated,
    extract_output_format,
    get_output_length
)

logger = logging.getLogger(__name__)


@dataclass
class ValidityFeatures:
    """Validity features for an LLM output."""

    is_well_formed: bool  # Can parse as expected format
    output_length: float  # Normalized length (log scale)
    has_error_stems: bool  # Contains error keywords
    format_matches_expectation: bool  # Format matches system prompt
    is_truncated: bool  # Output appears incomplete


def extract_expected_format_from_prompt(system_prompt: str) -> Optional[str]:
    """
    Detect expected output format from system prompt.

    Looks for phrases like:
    - "output JSON"
    - "return a dictionary"
    - "write Python code"
    - "respond with"

    Args:
        system_prompt: System prompt text

    Returns:
        Expected format: 'json', 'code', 'text', or None
    """
    if not system_prompt:
        return None

    prompt_lower = system_prompt.lower()

    # Check for JSON
    json_indicators = [
        'output json', 'return json', 'json object', 'json format',
        'output.*json', 'response.*json', 'must be json'
    ]
    for indicator in json_indicators:
        if re.search(indicator, prompt_lower):
            return 'json'

    # Check for code
    code_indicators = [
        'write code', 'python code', 'generate code', 'write a function',
        'implement', 'write python', 'code snippet'
    ]
    for indicator in code_indicators:
        if indicator in prompt_lower:
            return 'code'

    # Default to text
    return 'text'


def check_is_well_formed(output_text: str, expected_format: Optional[str]) -> bool:
    """
    Check if output is well-formed according to expected format.

    Args:
        output_text: LLM output text
        expected_format: Expected format ('json', 'code', 'text', or None)

    Returns:
        True if well-formed
    """
    if not output_text:
        return False

    if expected_format == 'json':
        # Try to parse as JSON
        _, is_valid = parse_json(output_text)
        return is_valid

    elif expected_format == 'code':
        # Extract code and check syntax
        code = extract_code_from_markdown(output_text)
        if code:
            return is_valid_python_code(code)
        else:
            # No code block, might be inline code
            return is_valid_python_code(output_text)

    elif expected_format == 'text':
        # Text is valid if it's non-empty and has reasonable structure
        return len(output_text.strip()) > 10

    else:
        # Unknown format, just check if non-empty
        return len(output_text.strip()) > 0


def normalize_output_length(length: int) -> float:
    """
    Normalize output length to reasonable scale.

    Uses log scale to handle wide range of lengths.

    Args:
        length: Raw output length

    Returns:
        Normalized length (0-10 scale)
    """
    if length <= 0:
        return 0.0

    # Log scale: log10(length)
    # Typical ranges:
    # - 10 chars -> 1.0
    # - 100 chars -> 2.0
    # - 1000 chars -> 3.0
    # - 10000 chars -> 4.0
    import math
    return min(math.log10(length + 1), 10.0)


def check_format_matches(
        output_text: str,
        expected_format: Optional[str]
) -> bool:
    """
    Check if output format matches expectation.

    Args:
        output_text: LLM output text
        expected_format: Expected format

    Returns:
        True if format matches
    """
    if not expected_format or not output_text:
        return True  # No expectation or no output

    actual_format = extract_output_format(output_text)

    # Allow some flexibility
    if expected_format == 'json':
        return actual_format in ['json', 'markdown']  # JSON can be in markdown
    elif expected_format == 'code':
        return actual_format in ['code', 'markdown']  # Code can be in markdown
    else:
        return True  # Text format is flexible


def extract_token_counts(task_data: Dict[str, Any]) -> Tuple[int, int, float]:
    """
    Extract token counts from task attributes.

    Args:
        task_data: Task data dictionary

    Returns:
        Tuple of (input_tokens, output_tokens, token_ratio)
    """
    attributes = task_data.attributes

    input_tokens = attributes.get('num_input_tokens', 0)
    output_tokens = attributes.get('num_output_tokens', 0)

    # Calculate ratio (output/input)
    if input_tokens > 0:
        token_ratio = output_tokens / input_tokens
    else:
        token_ratio = 0.0

    return int(input_tokens), int(output_tokens), float(token_ratio)


def compute_validity_features(
        task_data: Dict[str, Any],
        system_prompt: Optional[str] = None
) -> ValidityFeatures:
    """
    Compute all validity features for an LLM output.

    Args:
        task_data: Task data dictionary
        system_prompt: System prompt (for format expectation)

    Returns:
        ValidityFeatures object
    """
    # Extract output
    output_data = task_data.output
    output_text = output_data.get('gen_ai.completion.0.content', '')

    if not output_text:
        # No output - all features are negative
        return ValidityFeatures(
            is_well_formed=False,
            output_length=0.0,
            has_error_stems=False,
            format_matches_expectation=False,
            is_truncated=False
        )

    # Determine expected format
    expected_format = None
    if system_prompt:
        expected_format = extract_expected_format_from_prompt(system_prompt)

    # Compute features
    is_well_formed = check_is_well_formed(output_text, expected_format)

    raw_length = get_output_length(output_text)
    output_length = normalize_output_length(raw_length)

    has_errors = contains_error_stems(output_text)

    format_matches = check_format_matches(output_text, expected_format)

    truncated = is_truncated(output_text)

    return ValidityFeatures(
        is_well_formed=is_well_formed,
        output_length=output_length,
        has_error_stems=has_errors,
        format_matches_expectation=format_matches,
        is_truncated=truncated
    )


def validity_features_to_array(features: ValidityFeatures) -> np.ndarray:
    """
    Convert ValidityFeatures to numpy array.

    Args:
        features: ValidityFeatures object

    Returns:
        Numpy array of shape (5,)
    """
    return np.array([
        1.0 if features.is_well_formed else 0.0,
        features.output_length,
        1.0 if features.has_error_stems else 0.0,
        1.0 if features.format_matches_expectation else 0.0,
        1.0 if features.is_truncated else 0.0
    ], dtype=np.float32)


def compute_validity_features_batch(
        task_data_list: list[Dict[str, Any]],
        system_prompts: Optional[list[str]] = None
) -> np.ndarray:
    """
    Compute validity features for a batch of tasks.

    Args:
        task_data_list: List of task data dictionaries
        system_prompts: Optional list of system prompts (aligned with tasks)

    Returns:
        Numpy array of shape (n_tasks, 5)
    """
    features_list = []

    for i, task_data in enumerate(task_data_list):
        system_prompt = system_prompts[i] if system_prompts else None
        features = compute_validity_features(task_data, system_prompt)
        features_array = validity_features_to_array(features)
        features_list.append(features_array)

    return np.stack(features_list, axis=0)


def get_validity_feature_names() -> list[str]:
    """
    Get names of validity features.

    Returns:
        List of feature names
    """
    return [
        'is_well_formed',
        'output_length_normalized',
        'has_error_stems',
        'format_matches_expectation',
        'is_truncated'
    ]


def get_num_validity_features() -> int:
    """
    Get number of validity features.

    Returns:
        Number of validity features (5)
    """
    return 5
"""
Output encoding for LLM calls.

This module extracts and encodes LLM outputs, handling different
formats (JSON, code, text) and extracting semantic content.
"""

import logging
import re
import json
from typing import Dict, Any, Optional
import numpy as np
from utils.text_processing import (
    parse_json,
    extract_code_from_markdown,
    extract_output_format
)

logger = logging.getLogger(__name__)


def extract_llm_output(task_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract LLM output from task data.

    Looks for completion content in:
    - output['gen_ai.completion.0.content']

    Args:
        task_data: Task data dictionary

    Returns:
        LLM output text or None if not found
    """
    output_data = task_data.output

    # Try to get completion content
    output_text = output_data.get('gen_ai.completion.0.content')

    if output_text:
        return output_text

    # Fallback: look for any completion content
    for key, value in output_data.items():
        if 'completion' in key and 'content' in key:
            return value

    logger.warning("Could not extract LLM output from task")
    return None


def clean_llm_output(text: str, max_length: int = 5000) -> str:
    """
    Clean LLM output for encoding.

    Strategy:
    - For JSON: Extract key semantic fields
    - For code: Keep code but truncate if very long
    - For text: Clean and truncate

    Args:
        text: Raw LLM output
        max_length: Maximum length to keep

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    output_format = extract_output_format(text)

    # Handle JSON output
    if output_format == 'json':
        parsed, success = parse_json(text)
        if success and isinstance(parsed, dict):
            # Extract important fields
            semantic_fields = []

            # Common semantic fields
            for key in ['thoughts', 'reasoning', 'explanation', 'analysis',
                        'result', 'output', 'answer', 'response', 'content']:
                if key in parsed:
                    value = parsed[key]
                    if isinstance(value, (str, list)):
                        semantic_fields.append(str(value))

            if semantic_fields:
                text = ' '.join(semantic_fields)
            else:
                # Just use stringified JSON
                text = json.dumps(parsed)

    # Handle code output
    elif output_format in ['code', 'markdown']:
        # Try to extract code blocks
        code = extract_code_from_markdown(text)
        if code:
            text = code

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text.strip()


def encode_output(
        task_data: Dict[str, Any],
        embedding_model,
        max_length: int = 5000
) -> np.ndarray:
    """
    Encode the LLM output.

    Args:
        task_data: Task data dictionary
        embedding_model: EmbeddingModel instance
        max_length: Maximum length for output text

    Returns:
        Output embedding vector (embedding_dim,)
    """
    # Extract output
    output_text = extract_llm_output(task_data)

    if not output_text:
        logger.warning("No output found, returning zero vector")
        return np.zeros(embedding_model.dimension, dtype=np.float32)

    # Clean output
    cleaned_output = clean_llm_output(output_text, max_length)

    if not cleaned_output:
        logger.warning("Cleaned output is empty, returning zero vector")
        return np.zeros(embedding_model.dimension, dtype=np.float32)

    # Encode
    embedding = embedding_model.encode(cleaned_output)

    return embedding.astype(np.float32)


def encode_output_batch(
        task_data_list: list[Dict[str, Any]],
        embedding_model,
        max_length: int = 5000,
        batch_size: int = 32
) -> np.ndarray:
    """
    Encode multiple outputs in batch.

    Args:
        task_data_list: List of task data dictionaries
        embedding_model: EmbeddingModel instance
        max_length: Maximum length for output text
        batch_size: Batch size for encoding

    Returns:
        Output embeddings array (n_outputs, embedding_dim)
    """
    # Extract and clean all outputs
    cleaned_outputs = []
    for task_data in task_data_list:
        output_text = extract_llm_output(task_data)
        if output_text:
            cleaned = clean_llm_output(output_text, max_length)
            cleaned_outputs.append(cleaned if cleaned else "")
        else:
            cleaned_outputs.append("")

    # Encode in batch
    embeddings = embedding_model.encode(
        cleaned_outputs,
        batch_size=batch_size,
        show_progress=True
    )

    return embeddings.astype(np.float32)
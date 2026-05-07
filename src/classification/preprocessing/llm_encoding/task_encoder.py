"""
Task (user input) encoding for LLM calls.

This module extracts and encodes the user input portion of LLM calls,
handling variable-length inputs and extracting the core task description.
"""

import logging
import re
import json
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def extract_user_input(task_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract user input from LLM call task data.

    Looks for the user message in the input, typically in:
    - gen_ai.prompt.1.content (user message after system prompt)

    Args:
        task_data: Task data dictionary

    Returns:
        User input text or None if not found
    """
    input_data = task_data.input

    # Try to get user message (typically prompt.1)
    user_input = input_data.get('gen_ai.prompt.1.content')

    if user_input:
        return user_input

    # Fallback: look for any prompt with role 'user'
    for key, value in input_data.items():
        if 'prompt' in key and 'role' in key and value == 'user':
            # Get corresponding content
            content_key = key.replace('.role', '.content')
            if content_key in input_data:
                return input_data[content_key]

    logger.warning("Could not extract user input from task")
    return None


def clean_user_input(text: str, max_length: int = 5000) -> str:
    """
    Clean user input text for encoding.

    Removes:
    - Excessive JSON formatting
    - Very long data structures (keep descriptions, remove long lists)
    - Code blocks (keep intent, remove implementation)

    Args:
        text: Raw user input
        max_length: Maximum length to keep

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Try to parse as JSON
    try:
        # Handle JSON in markdown blocks
        if '```json' in text or '```' in text:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)

        # Try parsing
        parsed = json.loads(text)

        # Extract key fields (intent, task, goal, etc.)
        if isinstance(parsed, dict):
            key_fields = []

            # Look for intent/task/goal fields
            for key in ['intent', 'task', 'goal', 'objective', 'question', 'query']:
                if key in parsed:
                    key_fields.append(str(parsed[key]))

            # If we found key fields, use those
            if key_fields:
                text = ' '.join(key_fields)
            else:
                # Otherwise, just stringify (will be truncated if too long)
                text = json.dumps(parsed)

    except (json.JSONDecodeError, TypeError):
        # Not JSON or invalid, keep as is
        pass

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text.strip()


def encode_task(
        task_data: Dict[str, Any],
        embedding_model,
        max_length: int = 5000
) -> np.ndarray:
    """
    Encode the task (user input) from LLM call.

    Args:
        task_data: Task data dictionary
        embedding_model: EmbeddingModel instance
        max_length: Maximum length for input text

    Returns:
        Task embedding vector (embedding_dim,)
    """
    # Extract user input
    user_input = extract_user_input(task_data)

    if not user_input:
        logger.warning("No user input found, returning zero vector")
        return np.zeros(embedding_model.dimension, dtype=np.float32)

    # Clean input
    cleaned_input = clean_user_input(user_input, max_length)

    if not cleaned_input:
        logger.warning("Cleaned input is empty, returning zero vector")
        return np.zeros(embedding_model.dimension, dtype=np.float32)

    # Encode
    embedding = embedding_model.encode(cleaned_input)

    return embedding.astype(np.float32)


def encode_task_batch(
        task_data_list: list[Dict[str, Any]],
        embedding_model,
        max_length: int = 5000,
        batch_size: int = 32
) -> np.ndarray:
    """
    Encode multiple tasks in batch.

    Args:
        task_data_list: List of task data dictionaries
        embedding_model: EmbeddingModel instance
        max_length: Maximum length for input text
        batch_size: Batch size for encoding

    Returns:
        Task embeddings array (n_tasks, embedding_dim)
    """
    # Extract and clean all inputs
    cleaned_inputs = []
    for task_data in task_data_list:
        user_input = extract_user_input(task_data)
        if user_input:
            cleaned = clean_user_input(user_input, max_length)
            cleaned_inputs.append(cleaned if cleaned else "")
        else:
            cleaned_inputs.append("")

    # Encode in batch
    embeddings = embedding_model.encode(
        cleaned_inputs,
        batch_size=batch_size,
        show_progress=True
    )

    return embeddings.astype(np.float32)
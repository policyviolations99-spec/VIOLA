"""
LLM feature builder - assembles complete feature vectors for LLM nodes.

Combines:
- Role signature (32 dim)
- Task embedding (384/768 dim)
- Output embedding (384/768 dim)
- Token features (3 dim)
- Validity features (5 dim)
- Structural features (12 dim)

Total: 820-1588 dim depending on embedding model
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field

from llm_encoding.task_encoder import encode_task, encode_task_batch
from llm_encoding.output_encoder import encode_output, encode_output_batch, extract_llm_output
from llm_encoding.validity_features import (
    compute_validity_features,
    validity_features_to_array,
    extract_token_counts
)
from llm_encoding.structural_features import (
    extract_structural_features,
    STRUCTURAL_DIM,
    STRUCTURAL_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMFeatures:
    """Complete feature set for an LLM node."""

    role_signature: np.ndarray  # (32,)
    task_embedding: np.ndarray  # (embedding_dim,)
    output_embedding: np.ndarray  # (embedding_dim,)
    input_tokens: int
    output_tokens: int
    token_ratio: float
    validity_features: np.ndarray   # (5,)
    structural_features: np.ndarray = field(
        default_factory=lambda: np.zeros(STRUCTURAL_DIM, dtype=np.float32)
    )  # (12,)

    def to_array(self) -> np.ndarray:
        """
        Convert to single feature vector.

        Returns:
            Feature array of shape (total_dim,)
        """
        return np.concatenate([
            self.role_signature,
            self.task_embedding,
            self.output_embedding,
            np.array([self.input_tokens, self.output_tokens, self.token_ratio], dtype=np.float32),
            self.validity_features,
            self.structural_features,
        ])

    @property
    def dimension(self) -> int:
        """Get total feature dimension."""
        return len(self.to_array())


def extract_system_prompt(task_data: Dict[str, Any]) -> str:
    """
    Extract system prompt from task data.

    Args:
        task_data: Task data dictionary

    Returns:
        System prompt text (empty string if not found)
    """
    input_data = task_data.input
    system_prompt = input_data.get('gen_ai.prompt.0.content', '')
    return system_prompt


def build_llm_features(
        task_data: Dict[str, Any],
        prompt_cache,
        embedding_model
) -> LLMFeatures:
    """
    Build complete LLM feature vector for a single task.

    Args:
        task_data: Task data dictionary
        prompt_cache: PromptCache instance
        embedding_model: EmbeddingModel instance

    Returns:
        LLMFeatures object
    """
    # Extract system prompt
    system_prompt = extract_system_prompt(task_data)

    # Get role signature
    role_signature = prompt_cache.get_role_signature(system_prompt)

    # Encode task (user input)
    task_embedding = encode_task(task_data, embedding_model)

    # Encode output
    output_embedding = encode_output(task_data, embedding_model)

    # Extract token counts
    input_tokens, output_tokens, token_ratio = extract_token_counts(task_data)

    # Compute validity features
    validity_features_obj = compute_validity_features(task_data, system_prompt)
    validity_features = validity_features_to_array(validity_features_obj)

    # Compute structural features from raw JSON output
    output_text = extract_llm_output(task_data)
    structural = extract_structural_features(output_text)

    return LLMFeatures(
        role_signature=role_signature,
        task_embedding=task_embedding,
        output_embedding=output_embedding,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_ratio=token_ratio,
        validity_features=validity_features,
        structural_features=structural,
    )


def build_llm_features_batch(
        task_data_list: List[Dict[str, Any]],
        prompt_cache,
        embedding_model,
        batch_size: int = 32
) -> np.ndarray:
    """
    Build LLM features for a batch of tasks.

    Args:
        task_data_list: List of task data dictionaries
        prompt_cache: PromptCache instance
        embedding_model: EmbeddingModel instance
        batch_size: Batch size for encoding

    Returns:
        Feature array of shape (n_tasks, total_dim)
    """
    logger.info(f"Building LLM features for {len(task_data_list)} tasks...")

    # Extract system prompts
    system_prompts = [extract_system_prompt(task_data) for task_data in task_data_list]

    # Get role signatures
    logger.info("  - Extracting role signatures...")
    role_signatures = np.stack([
        prompt_cache.get_role_signature(prompt)
        for prompt in system_prompts
    ])

    # Encode tasks in batch
    logger.info("  - Encoding task inputs...")
    task_embeddings = encode_task_batch(task_data_list, embedding_model, batch_size=batch_size)

    # Encode outputs in batch
    logger.info("  - Encoding outputs...")
    output_embeddings = encode_output_batch(task_data_list, embedding_model, batch_size=batch_size)

    # Extract token features
    logger.info("  - Extracting token features...")
    token_features = []
    for task_data in task_data_list:
        input_tokens, output_tokens, token_ratio = extract_token_counts(task_data)
        token_features.append([input_tokens, output_tokens, token_ratio])
    token_features = np.array(token_features, dtype=np.float32)

    # Compute validity features
    logger.info("  - Computing validity features...")
    validity_features_batch = []
    for i, task_data in enumerate(task_data_list):
        validity_features_obj = compute_validity_features(task_data, system_prompts[i])
        validity_features = validity_features_to_array(validity_features_obj)
        validity_features_batch.append(validity_features)
    validity_features_batch = np.stack(validity_features_batch)

    # Compute structural features
    logger.info("  - Extracting structural features...")
    structural_batch = np.stack([
        extract_structural_features(extract_llm_output(td))
        for td in task_data_list
    ])

    # Concatenate all features
    logger.info("  - Concatenating features...")
    all_features = np.concatenate([
        role_signatures,
        task_embeddings,
        output_embeddings,
        token_features,
        validity_features_batch,
        structural_batch,
    ], axis=1)

    logger.info(f"✓ Built LLM features with shape: {all_features.shape}")

    return all_features


def get_llm_feature_dimension(embedding_dim: int, role_signature_dim: int = 32) -> int:
    """
    Calculate total LLM feature dimension.

    Args:
        embedding_dim: Embedding dimension (384 or 768)
        role_signature_dim: Role signature dimension

    Returns:
        Total feature dimension
    """
    return (
            role_signature_dim +  # 32
            embedding_dim +  # 384/768 (task)
            embedding_dim +  # 384/768 (output)
            3 +  # token features
            5 +  # validity features
            STRUCTURAL_DIM  # 12 structural features
    )


def get_llm_feature_names(embedding_dim: int, role_signature_dim: int = 32) -> List[str]:
    """
    Get names of all LLM features.

    Args:
        embedding_dim: Embedding dimension
        role_signature_dim: Role signature dimension

    Returns:
        List of feature names
    """
    names = []

    # Role signature
    names.extend([f'role_sig_{i}' for i in range(role_signature_dim)])

    # Task embedding
    names.extend([f'task_emb_{i}' for i in range(embedding_dim)])

    # Output embedding
    names.extend([f'output_emb_{i}' for i in range(embedding_dim)])

    # Token features
    names.extend(['input_tokens', 'output_tokens', 'token_ratio'])

    # Validity features
    names.extend([
        'is_well_formed',
        'output_length_normalized',
        'has_error_stems',
        'format_matches_expectation',
        'is_truncated'
    ])

    # Structural features
    names.extend(STRUCTURAL_FEATURE_NAMES)

    return names
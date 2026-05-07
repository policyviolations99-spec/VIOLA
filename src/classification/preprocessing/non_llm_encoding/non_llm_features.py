"""
Non-LLM node feature extraction.

Non-LLM nodes use zero-padded embeddings since they don't have semantic content.
All meaningful features are in common_features (hierarchy, duration, name, tags).
"""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


def build_non_llm_features(
    node_info,
    target_dim: int
) -> np.ndarray:
    """
    Build feature vector for a non-LLM node.

    Returns zero-padded vector since non-LLM nodes don't have:
    - Role signatures (no system prompts)
    - Task embeddings (no user inputs)
    - Output embeddings (no LLM outputs)
    - Token counts (no LLM tokens)
    - Validity features (no LLM outputs to validate)

    All meaningful features are captured in common_features.

    Args:
        node_info: NodeInfo object
        target_dim: Target dimension to match LLM features

    Returns:
        Zero feature vector of shape (target_dim,)
    """
    return np.zeros(target_dim, dtype=np.float32)


def build_non_llm_features_batch(
    node_infos: List,
    target_dim: int
) -> np.ndarray:
    """
    Build features for a batch of non-LLM nodes.

    Args:
        node_infos: List of NodeInfo objects
        target_dim: Target dimension to match LLM features

    Returns:
        Zero feature array of shape (n_nodes, target_dim)
    """
    if not node_infos:
        return np.zeros((0, target_dim), dtype=np.float32)

    logger.info(f"Building zero-padded features for {len(node_infos)} non-LLM nodes...")

    features = np.zeros((len(node_infos), target_dim), dtype=np.float32)

    logger.info(f"✓ Built non-LLM features with shape: {features.shape}")

    return features
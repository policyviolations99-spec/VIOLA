"""
Common features shared by all node types.

This module computes features that apply to all nodes in the execution trace,
regardless of whether they are LLM calls or other node types.

Updated to include:
- Hierarchy information from name prefix
- Node name embeddings
- Task type flags from tags
- Binary node types (llm vs non-llm)
"""

import logging
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from src.preprocessing.utils.embeddings import NameEmbedder

logger = logging.getLogger(__name__)


@dataclass
class CommonFeatures:
    """Common features for a node."""

    duration: float  # Duration in seconds (or -1 if missing)
    node_type_id: int  # Binary node type (0=llm, 1=non-llm)
    hierarchy_depth: int  # Depth in execution tree (from name prefix)
    sibling_index: int  # Position among siblings (from name prefix)
    parent_index: int  # Parent's sibling index (from name prefix)
    name_embedding: np.ndarray  # Embedding of node name (32-dim)
    task_type_flags: np.ndarray  # Binary flags for task types (3-dim)


# Node type mapping - BINARY (llm vs non-llm)
NODE_TYPE_MAPPING = {
    'llm_call': 0,
    'non_llm': 1
}

# Task type tags (non-framework tags)
TASK_TYPE_TAGS = ['llm_call', 'tool_call', 'complex']


def parse_name_prefix(name: str) -> Tuple[str, str]:
    """
    Parse name into prefix and suffix.

    Example: "0.3.2:PlanControllerAgent" → ("0.3.2", "PlanControllerAgent")

    Args:
        name: Full node name

    Returns:
        Tuple of (prefix, suffix)
    """
    if ':' in name:
        parts = name.split(':', 1)
        return parts[0], parts[1]
    else:
        return "", name


def extract_hierarchy_info(prefix: str) -> Tuple[int, int, int]:
    """
    Extract hierarchy information from name prefix.

    Example: "0.3.2" → depth=3, sibling_index=2, parent_index=3

    Args:
        prefix: Name prefix (e.g., "0.3.2")

    Returns:
        Tuple of (depth, sibling_index, parent_index)
    """
    if not prefix:
        return 0, 0, 0

    parts = prefix.split('.')

    depth = len(parts)
    sibling_index = int(parts[-1]) if parts else 0
    parent_index = int(parts[-2]) if len(parts) >= 2 else 0

    return depth, sibling_index, parent_index


def extract_task_type_flags(tags: List[str]) -> np.ndarray:
    """
    Extract binary flags for task types from tags.

    Only considers non-framework tags: llm_call, tool_call, complex
    Ignores framework tags like: LangGraph Node, etc.

    Args:
        tags: List of tag strings

    Returns:
        Binary flag array of shape (3,): [is_llm, is_tool, is_complex]
    """
    flags = np.zeros(3, dtype=np.float32)

    if not tags:
        return flags

    # Convert to lowercase for case-insensitive matching
    tags_lower = [tag.lower() for tag in tags]

    flags[0] = 1.0 if 'llm_call' in tags_lower else 0.0
    flags[1] = 1.0 if 'tool_call' in tags_lower else 0.0
    flags[2] = 1.0 if 'complex' in tags_lower else 0.0

    return flags


def compute_duration_feature(node_info) -> float:
    """
    Compute duration feature for a node.

    Args:
        node_info: NodeInfo object

    Returns:
        Duration in seconds, or -1.0 if not available
    """
    if node_info.duration is not None and node_info.duration >= 0:
        return float(node_info.duration)
    else:
        return -1.0


def compute_node_type_id(node_info) -> int:
    """
    Compute node type ID (binary: 0=llm, 1=non-llm).

    Args:
        node_info: NodeInfo object

    Returns:
        Node type ID (0 or 1)
    """
    return NODE_TYPE_MAPPING.get(node_info.node_type, 1)  # Default to 'non_llm'


def compute_common_features(node_info, name_embedder=None) -> CommonFeatures:
    """
    Compute all common features for a node.

    Args:
        node_info: NodeInfo object
        name_embedder: Optional NameEmbedder for encoding node names

    Returns:
        CommonFeatures object
    """
    # Duration and node type
    duration = compute_duration_feature(node_info)
    node_type_id = compute_node_type_id(node_info)

    # Parse name
    task_data = node_info.task_data
    name = task_data.name
    prefix, suffix = parse_name_prefix(name)

    # Extract hierarchy info
    depth, sibling_idx, parent_idx = extract_hierarchy_info(prefix)

    # Extract task type flags
    tags = task_data.tags
    task_type_flags = extract_task_type_flags(tags)

    # Encode name
    if name_embedder:
        name_embedding = name_embedder.encode(suffix)
    else:
        name_embedding = np.zeros(32, dtype=np.float32)

    return CommonFeatures(
        duration=duration,
        node_type_id=node_type_id,
        hierarchy_depth=depth,
        sibling_index=sibling_idx,
        parent_index=parent_idx,
        name_embedding=name_embedding,
        task_type_flags=task_type_flags
    )


def common_features_to_array(features: CommonFeatures) -> np.ndarray:
    """
    Convert CommonFeatures to numpy array.

    Args:
        features: CommonFeatures object

    Returns:
        Numpy array of shape (40,)
        [duration, node_type_id, depth, sibling_idx, parent_idx,
         name_embedding(32), task_type_flags(3)]
    """
    return np.concatenate([
        np.array([
            features.duration,
            features.node_type_id,
            features.hierarchy_depth,
            features.sibling_index,
            features.parent_index
        ], dtype=np.float32),
        features.name_embedding,
        features.task_type_flags
    ])


def compute_common_features_batch(node_infos: List, name_embedder=None) -> np.ndarray:
    """
    Compute common features for a batch of nodes.

    Args:
        node_infos: List of NodeInfo objects
        name_embedder: Optional NameEmbedder for encoding node names

    Returns:
        Numpy array of shape (n_nodes, 40)
    """
    features_list = []
    for node_info in node_infos:
        features = compute_common_features(node_info, name_embedder)
        features_array = common_features_to_array(features)
        features_list.append(features_array)

    return np.stack(features_list, axis=0)


def get_common_feature_names() -> List[str]:
    """
    Get names of common features.

    Returns:
        List of feature names
    """
    names = [
        'duration',
        'node_type_id',
        'hierarchy_depth',
        'sibling_index',
        'parent_index'
    ]
    names.extend([f'name_emb_{i}' for i in range(32)])
    names.extend(['is_llm_call', 'is_tool_call', 'is_complex'])
    return names


def get_num_common_features() -> int:
    """
    Get number of common features.

    Returns:
        Number of common features (40)
    """
    return 40


def build_name_embedder(all_nodes: List) -> NameEmbedder:
    """
    Build a name embedder from all nodes.

    Args:
        all_nodes: List of all NodeInfo objects

    Returns:
        Fitted NameEmbedder
    """
    logger.info("Building name embedder...")

    # Extract all unique names
    names = set()
    for node in all_nodes:
        task_data = node.task_data
        name = task_data.name
        if name:
            _, suffix = parse_name_prefix(name)
            if suffix:
                names.add(suffix)

    names = list(names)
    logger.info(f"Found {len(names)} unique node names")

    # Build embedder
    embedder = NameEmbedder(embedding_dim=32)
    embedder.fit(names)

    return embedder

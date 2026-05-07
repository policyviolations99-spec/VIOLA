"""
Extract nodes and edges from execution trace tasks.

This module parses task dictionaries to:
1. Identify node types (LLM calls vs non-LLM nodes)
2. Extract parent-child relationships for graph edges
3. Build node ID mappings for sequential indexing
"""

import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a single node in the execution trace."""

    task_id: str  # Original task ID
    node_idx: int  # Sequential node index (0, 1, 2, ...)
    node_type: str  # 'llm_call' or 'non_llm'
    parent_id: str | None  # Parent task ID (None for root)
    parent_idx: int | None  # Parent node index (None for root)

    # Timing
    start_time: datetime | None
    end_time: datetime | None
    duration: float | None  # in seconds

    # Raw task data for later processing
    task_data: Dict[str, Any]


def parse_datetime(dt_str: str | None) -> datetime | None:
    """
    Parse datetime string to datetime object.

    Args:
        dt_str: ISO format datetime string or datetime object

    Returns:
        datetime object or None if parsing fails
    """
    if dt_str is None:
        return None

    # If already a datetime object, return it
    if isinstance(dt_str, datetime):
        return dt_str

    try:
        # Handle ISO format with 'Z' suffix
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1] + '+00:00'
        return datetime.fromisoformat(dt_str)
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse datetime: {dt_str} - {e}")
        return None


def calculate_duration(start_time: datetime | None, end_time: datetime | None) -> float | None:
    """
    Calculate duration in seconds between start and end time.

    Args:
        start_time: Start datetime
        end_time: End datetime

    Returns:
        Duration in seconds or None if times are invalid
    """
    if start_time is None or end_time is None:
        return None

    try:
        delta = end_time - start_time
        return delta.total_seconds()
    except Exception as e:
        logger.warning(f"Failed to calculate duration: {e}")
        return None


def identify_node_type(task: Dict[str, Any]) -> str:
    """
    Identify the type of node based on task tags.
    
    Simplified to binary classification:
    - 'llm_call': LLM nodes
    - 'non_llm': Everything else (tool_call, complex, etc.)

    Args:
        task: Task dictionary

    Returns:
        Node type: 'llm_call' or 'non_llm'
    """
    tags = task.tags

    if 'llm_call' in tags:
        return 'llm_call'
    else:
        return 'non_llm'


def extract_nodes_and_edges(
        tasks_dict: Dict[str, Dict[str, Any]]
) -> Tuple[List[NodeInfo], List[Tuple[int, int]]]:
    """
    Extract node information and edges from tasks dictionary.

    Args:
        tasks_dict: Dictionary mapping task IDs to task data

    Returns:
        Tuple of (nodes, edges) where:
        - nodes: List of NodeInfo objects
        - edges: List of (parent_idx, child_idx) tuples
    """
    logger.info(f"Extracting nodes and edges from {len(tasks_dict)} tasks")

    # Build node ID mapping: task_id -> sequential index
    task_ids = list(tasks_dict.keys())
    id_to_idx = {task_id: idx for idx, task_id in enumerate(task_ids)}

    # Extract node information
    nodes = []
    for task_id in task_ids:
        task = tasks_dict[task_id]
        node_idx = id_to_idx[task_id]

        # Get parent information
        parent_id = task.parent_id
        parent_idx = id_to_idx.get(parent_id) if parent_id else None

        # Parse timing
        start_time = parse_datetime(task.start_time)
        end_time = parse_datetime(task.end_time)
        duration = calculate_duration(start_time, end_time)

        # Identify node type
        node_type = identify_node_type(task)

        # Create node info
        node = NodeInfo(
            task_id=task_id,
            node_idx=node_idx,
            node_type=node_type,
            parent_id=parent_id,
            parent_idx=parent_idx,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            task_data=task
        )
        nodes.append(node)

    # Extract edges (parent -> child relationships)
    edges = []
    for node in nodes:
        if node.parent_idx is not None:
            edges.append((node.parent_idx, node.node_idx))

    # Log statistics
    node_type_counts = {}
    for node in nodes:
        node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1

    logger.info(f"✓ Extracted {len(nodes)} nodes:")
    for node_type, count in node_type_counts.items():
        logger.info(f"  - {node_type}: {count}")
    logger.info(f"✓ Extracted {len(edges)} edges")

    # Validation
    root_nodes = [n for n in nodes if n.parent_idx is None]
    logger.info(f"  - Root nodes (no parent): {len(root_nodes)}")

    if len(root_nodes) == 0:
        logger.warning("No root nodes found - graph may be disconnected or have cycles")
    elif len(root_nodes) > 1:
        logger.warning(f"Multiple root nodes found ({len(root_nodes)}) - expected single root for tree")

    return nodes, edges


def group_nodes_by_type(nodes: List[NodeInfo]) -> Dict[str, List[NodeInfo]]:
    """
    Group nodes by their type.

    Args:
        nodes: List of NodeInfo objects

    Returns:
        Dictionary mapping node type to list of nodes
    """
    grouped = {}
    for node in nodes:
        if node.node_type not in grouped:
            grouped[node.node_type] = []
        grouped[node.node_type].append(node)

    return grouped


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

"""
PyTorch Geometric data converter.

Converts preprocessed features and graph structure into PyG Data objects
for GNN training.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def convert_to_pyg_data(
        llm_features: np.ndarray,
        non_llm_features: np.ndarray,
        common_features: np.ndarray,
        edges: List[Tuple[int, int]],
        node_types: np.ndarray
) -> Data:
    """
    Convert preprocessed data to PyTorch Geometric Data object.

    Args:
        llm_features: LLM node features (n_llm, llm_dim)
        non_llm_features: Non-LLM node features (n_non_llm, llm_dim) [zero-padded]
        common_features: Common features for all nodes (n_nodes, 2)
        edges: List of (parent_idx, child_idx) tuples
        node_types: Node type IDs (n_nodes,)

    Returns:
        PyTorch Geometric Data object
    """
    # Concatenate all node features
    # LLM nodes come first, then non-LLM nodes
    all_node_features = np.vstack([llm_features, non_llm_features])

    # Add common features (duration, node_type_id)
    all_node_features = np.hstack([all_node_features, common_features])

    # Convert to tensors
    x = torch.tensor(all_node_features, dtype=torch.float32)

    # Convert edges to edge_index format
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # No edges (single node graph)
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Node types
    node_type = torch.tensor(node_types, dtype=torch.long)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        node_type=node_type,
        num_nodes=len(all_node_features)
    )

    return data


def save_pyg_data(data: Data, filepath: Path) -> None:
    """
    Save PyG Data object to disk.

    Args:
        data: PyTorch Geometric Data object
        filepath: Path to save to (.pt file)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, filepath)
    logger.info(f"✓ Saved PyG data to {filepath}")


def load_pyg_data(filepath: Path) -> Data:
    """
    Load PyG Data object from disk.

    Args:
        filepath: Path to load from (.pt file)

    Returns:
        PyTorch Geometric Data object
    """
    data = torch.load(filepath)
    logger.info(f"✓ Loaded PyG data from {filepath}")
    return data


def get_data_statistics(data: Data) -> Dict[str, Any]:
    """
    Get statistics about a PyG Data object.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.shape[1],
        'feature_dim': data.x.shape[1],
        'node_types': {
            'llm_call': (data.node_type == 0).sum().item(),
            'tool_call': (data.node_type == 1).sum().item(),
            'other': (data.node_type == 2).sum().item()
        },
        'feature_ranges': {
            'min': data.x.min().item(),
            'max': data.x.max().item(),
            'mean': data.x.mean().item(),
            'std': data.x.std().item()
        }
    }
    return stats


def print_data_statistics(data: Data) -> None:
    """
    Print statistics about a PyG Data object.

    Args:
        data: PyTorch Geometric Data object
    """
    stats = get_data_statistics(data)

    print("\n" + "=" * 60)
    print("PyG Data Statistics")
    print("=" * 60)
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Feature dimension: {stats['feature_dim']}")
    print(f"\nNode types:")
    for node_type, count in stats['node_types'].items():
        print(f"  - {node_type}: {count}")
    print(f"\nFeature ranges:")
    print(f"  - Min: {stats['feature_ranges']['min']:.4f}")
    print(f"  - Max: {stats['feature_ranges']['max']:.4f}")
    print(f"  - Mean: {stats['feature_ranges']['mean']:.4f}")
    print(f"  - Std: {stats['feature_ranges']['std']:.4f}")
    print("=" * 60 + "\n")
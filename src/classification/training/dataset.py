"""
Dataset and DataLoader for preprocessed execution traces.

Loads batch files saved by the preprocessing pipeline and handles
train/val/test splitting with optional label integration.

Supports two label modes:
  - binary:     single y label (original success/failure)
  - multi_task: y_agent (agent id) + y_failure (failure type id)
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data, Batch as PyGBatch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import logging
import json
import re

# Import from preprocessing (label_utils moved there)
import sys
_ROOT = Path(__file__).resolve().parent.parent.parent  # pattern-analysis/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.preprocessing.label_utils import (
    load_labels_from_results_csv,
    load_labels_json,
    load_labels_from_manifest_csv,
)

logger = logging.getLogger(__name__)


def natural_sort_key(s):
    """
    Key function for natural sorting of strings with numbers.
    
    Example: trace1.log, trace2.log, ..., trace10.log, trace11.log
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


class TraceDataset(Dataset):
    """
    Dataset for preprocessed execution traces.

    Supports two label modes:
      - 'binary':     single y label (original success/failure)
      - 'multi_task': y_agent (int) + y_failure (int), loaded from manifest.csv

    Label lookup uses data.log_filename (stored by preprocessing pipeline) when
    available, falling back to synthetic trace{N}.log names for backward compat.
    """

    def __init__(
        self,
        data_dir: Path,
        labels_file: Optional[Path] = None,
        label_mapping: Optional[Dict] = None,
        log_dir: Optional[Path] = None,
        manifest_file: Optional[Path] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir:      Directory with batch_*.pt files
            labels_file:   Path to labels JSON file (binary mode)
            label_mapping: Pre-loaded label dict (binary or multi-task)
            log_dir:       Path to raw log files (for results.csv, binary mode)
            manifest_file: Path to manifest.csv (multi-task mode)
        """
        self.data_dir = Path(data_dir)
        self.labels_file = Path(labels_file) if labels_file else None
        self.log_dir = Path(log_dir) if log_dir else None
        self.manifest_file = Path(manifest_file) if manifest_file else None

        # ── label loading ──────────────────────────────────────────────────
        if manifest_file is not None:
            # multi_task mode: labels[filename] = {'agent': int, 'violation': int}
            self.label_mode = 'multi_task'
            self.labels, self.agent_mapping, self.violation_mapping = \
                load_labels_from_manifest_csv(self.manifest_file)
            # Keep failure_mapping as alias for backward compatibility
            self.failure_mapping = self.violation_mapping
            logger.info(f"Multi-task mode: {len(self.agent_mapping)} agents, "
                        f"{len(self.violation_mapping)} violation types")
        elif label_mapping is not None:
            self.label_mode = 'binary'
            self.labels = label_mapping
            self.agent_mapping = {}
            self.violation_mapping = {}
            self.failure_mapping = {}
            logger.info("Using provided label mapping (binary mode)")
        elif self.log_dir:
            self.label_mode = 'binary'
            logger.info(f"Loading labels from results.csv in: {self.log_dir}")
            self.labels, _ = load_labels_from_results_csv(self.log_dir, batch_size=100)
            self.agent_mapping = {}
            self.violation_mapping = {}
            self.failure_mapping = {}
        elif self.labels_file and self.labels_file.exists():
            self.label_mode = 'binary'
            self.labels = load_labels_json(self.labels_file)
            self.agent_mapping = {}
            self.violation_mapping = {}
            self.failure_mapping = {}
        else:
            raise ValueError(
                "No labels provided! Supply one of: manifest_file, labels_file, "
                "label_mapping, or log_dir"
            )

        logger.info(f"Loaded {len(self.labels)} labels ({self.label_mode} mode)")
        available_traces = set(self.labels.keys())

        # ── load Data objects ──────────────────────────────────────────────
        self.data_list: List[Data] = []
        self.log_filenames: List[str] = []

        batch_files = sorted(self.data_dir.glob("batch_*.pt"))
        if not batch_files:
            raise ValueError(f"No batch files found in {data_dir}")

        logger.info(f"Loading data from {len(batch_files)} batch files...")

        for batch_file in batch_files:
            batch_data = torch.load(batch_file, weights_only=False)

            if not isinstance(batch_data, list):
                raise ValueError(f"Unexpected batch format in {batch_file}")

            batch_idx = int(batch_file.stem.split('_')[1])
            start_idx = batch_idx * 100  # fallback synthetic naming

            for i, data in enumerate(batch_data):
                # Prefer filename stored by preprocessing; fall back to trace{N}.log
                if hasattr(data, 'log_filename') and data.log_filename:
                    log_filename = data.log_filename
                else:
                    log_filename = f"trace{start_idx + i + 1}.log"

                if log_filename in available_traces:
                    self.data_list.append(data)
                    self.log_filenames.append(log_filename)
                else:
                    logger.debug(f"Skipping {log_filename} - no label available")

        logger.info(f"Loaded {len(self.data_list)} traces (filtered by available labels)")
        if self.log_filenames:
            logger.info(f"First: {self.log_filenames[0]}  Last: {self.log_filenames[-1]}")

        missing = available_traces - set(self.log_filenames)
        if missing:
            logger.warning(
                f"{len(missing)} traces have labels but no data: "
                f"{sorted(missing)[:5]}"
            )

        if self.data_list:
            first = self.data_list[0]
            logger.info(f"Feature dim: {first.x.shape[1]}  "
                        f"Avg nodes: {np.mean([d.num_nodes for d in self.data_list]):.1f}")

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        data = self.data_list[idx].clone()
        log_filename = self.log_filenames[idx]
        label = self.labels[log_filename]

        if self.label_mode == 'multi_task':
            data.y_agent = torch.tensor(label['agent'], dtype=torch.long)
            # Support both 'violation' (new) and 'failure' (legacy) key
            violation_label = label.get('violation', label.get('failure'))
            data.y_failure = torch.tensor(violation_label, dtype=torch.long)
        else:
            data.y = torch.tensor(label, dtype=torch.float)

        return data

    # ──────────────────────────────────────────────────────────────────────
    def get_class_distribution(self) -> Dict[int, int]:
        """Return label distribution (binary mode only)."""
        distribution: Dict[int, int] = {}
        for log_filename in self.log_filenames:
            label = self.labels[log_filename]
            if self.label_mode == 'multi_task':
                violation_label = label.get('violation', label.get('failure'))
                key = label['agent'] * 100 + violation_label  # combined key
            else:
                key = int(label)
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def get_multitask_distribution(self) -> Dict[Tuple, int]:
        """Return (agent, violation) pair distribution (multi-task mode only)."""
        distribution: Dict[Tuple, int] = {}
        for log_filename in self.log_filenames:
            label = self.labels[log_filename]
            if self.label_mode == 'multi_task':
                violation_label = label.get('violation', label.get('failure'))
                key = (label['agent'], violation_label)
            else:
                key = (int(label),)
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def get_trace_statistics(self) -> Dict[str, float]:
        """
        Get statistics about traces.

        Returns:
            Dictionary of statistics
        """
        num_nodes = [data.num_nodes for data in self.data_list]
        num_edges = [data.edge_index.shape[1] for data in self.data_list]

        return {
            'num_traces': len(self.data_list),
            'avg_nodes': np.mean(num_nodes),
            'std_nodes': np.std(num_nodes),
            'min_nodes': np.min(num_nodes),
            'max_nodes': np.max(num_nodes),
            'avg_edges': np.mean(num_edges),
            'std_edges': np.std(num_edges),
            'min_edges': np.min(num_edges),
            'max_edges': np.max(num_edges)
        }


def split_dataset(
    dataset: TraceDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train/val/test with stratification.

    For multi-task datasets, stratifies by the (agent, failure) pair so every
    combination is represented in each split. For binary datasets, stratifies
    by the single class label.

    Args:
        dataset: TraceDataset instance
        train_ratio: Fraction for training
        val_ratio:   Fraction for validation
        test_ratio:  Fraction for testing
        seed:        Random seed for reproducibility

    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)

    # ── build per-class index lists ────────────────────────────────────────
    class_to_indices: Dict[tuple, List[int]] = {}
    for idx, log_filename in enumerate(dataset.log_filenames):
        label = dataset.labels[log_filename]
        if dataset.label_mode == 'multi_task':
            violation_label = label.get('violation', label.get('failure'))
            key = (label['agent'], violation_label)
        else:
            key = (int(label),)
        class_to_indices.setdefault(key, []).append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for key, indices in sorted(class_to_indices.items()):
        indices = rng.permutation(indices).tolist()
        n = len(indices)

        n_train = max(1, round(train_ratio * n))
        n_val   = max(1, round(val_ratio * n))
        # test gets the remainder (guarantees at least 1 if n >= 3)
        n_test  = n - n_train - n_val

        if n_test < 0:
            # With very small n, give priority: train > val > test
            n_val  = max(0, n - n_train - 1)
            n_test = n - n_train - n_val

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    logger.info(
        f"Stratified split — train: {len(train_indices)}, "
        f"val: {len(val_indices)}, test: {len(test_indices)}"
    )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def custom_collate_fn(batch):
    """
    Custom collate function for PyG Data objects.

    Args:
        batch: List of Data objects (each with y attribute)

    Returns:
        Batched PyG Data object with y attached
    """
    # Batch PyG Data objects (automatically batches y as well)
    batched_data = PyGBatch.from_data_list(batch)

    return batched_data


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyG dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader


def compute_pos_weight(dataset: TraceDataset) -> float:
    """
    Compute positive class weight for handling class imbalance.

    pos_weight = num_negative / num_positive

    Args:
        dataset: TraceDataset instance

    Returns:
        Positive class weight
    """
    class_dist = dataset.get_class_distribution()

    num_positive = class_dist.get(1, 0)
    num_negative = class_dist.get(0, 0)

    if num_positive == 0:
        logger.warning("No positive samples found, setting pos_weight=1.0")
        return 1.0

    pos_weight = num_negative / num_positive

    logger.info(f"Class distribution: {class_dist}")
    logger.info(f"Computed pos_weight: {pos_weight:.2f}")

    return pos_weight


def print_dataset_statistics(dataset: TraceDataset):
    """Print dataset statistics."""
    logger.info("\n" + "=" * 80)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 80)

    stats = dataset.get_trace_statistics()
    logger.info(f"\nTrace Statistics:")
    logger.info(f"  - Total traces: {stats['num_traces']}")
    logger.info(f"  - Avg nodes: {stats['avg_nodes']:.1f} ± {stats['std_nodes']:.1f}")
    logger.info(f"  - Node range: [{stats['min_nodes']}, {stats['max_nodes']}]")
    logger.info(f"  - Avg edges: {stats['avg_edges']:.1f} ± {stats['std_edges']:.1f}")
    logger.info(f"  - Edge range: [{stats['min_edges']}, {stats['max_edges']}]")

    if dataset.label_mode == 'multi_task':
        dist = dataset.get_multitask_distribution()
        logger.info(f"\n(agent, violation) Distribution:")
        for (agent, violation), count in sorted(dist.items()):
            pct = 100.0 * count / stats['num_traces']
            logger.info(f"  - agent={agent}, violation={violation}: {count} ({pct:.1f}%)")
    else:
        class_dist = dataset.get_class_distribution()
        logger.info(f"\nClass Distribution:")
        for label, count in sorted(class_dist.items()):
            pct = 100.0 * count / stats['num_traces']
            logger.info(f"  - Class {label}: {count} ({pct:.1f}%)")

    logger.info("=" * 80 + "\n")
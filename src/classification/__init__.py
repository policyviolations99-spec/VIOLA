"""
GNN-based policy violation classifier.

Two-stage pipeline:
  1. Preprocessing: convert raw OTel traces to PyG graph objects with node features
  2. Training: train TraceGNN (heterogeneous GAT) for multi-task classification
     - Head 1: which agent was injected (5-class)
     - Head 2: which violation type was injected (12-class)
"""

from .training.model import TraceGNN
from .training.trainer import Trainer
from .training.config import TrainingConfig

__all__ = ["TraceGNN", "Trainer", "TrainingConfig"]

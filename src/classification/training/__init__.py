"""
Training package for GNN trace analysis.

This package provides a complete training pipeline for Graph Neural Networks
on execution traces with binary classification for success/failure prediction.
"""

from .config import Config, get_default_config, get_small_config, get_large_config
from .model import TraceGNN, count_parameters, print_model_summary
from .dataset import TraceDataset, split_dataset, create_dataloaders
from .trainer import Trainer, MultiTaskMetricsTracker
from .visualization import generate_all_plots

__version__ = '1.0.0'

__all__ = [
    'Config',
    'get_default_config',
    'get_small_config',
    'get_large_config',
    'TraceGNN',
    'count_parameters',
    'print_model_summary',
    'TraceDataset',
    'split_dataset',
    'create_dataloaders',
    'Trainer',
    'MultiTaskMetricsTracker',
    'generate_all_plots',
]

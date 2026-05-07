#!/usr/bin/env python
"""
Ready-to-run experiment script for cluster jobs.

This script allows you to run multiple experiments with different configurations.
Each experiment is tracked with a unique ID and all results are organized.

Usage:
    python run_experiment.py --experiment baseline
    python run_experiment.py --experiment large_model
    python run_experiment.py --experiment custom --hidden-dim 384 --num-layers 4
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

from main import train_model
from config import Config, get_default_config, get_small_config, get_large_config


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENTS = {
    # Baseline experiments
    "baseline": {
        "description": "Default configuration baseline",
        "config": "default",
        "params": {}
    },
    
    "baseline_small": {
        "description": "Small model for quick testing",
        "config": "small",
        "params": {
            "num_epochs": 100
        }
    },
    
    "baseline_large": {
        "description": "Large model for maximum performance",
        "config": "large",
        "params": {
            "num_epochs": 300
        }
    },
    
    # Architecture variations
    "deep_network": {
        "description": "Deeper network (4 layers)",
        "config": "default",
        "params": {
            "num_layers": 4,
            "num_epochs": 250
        }
    },
    
    "wide_network": {
        "description": "Wider network (hidden_dim=512)",
        "config": "default",
        "params": {
            "hidden_dim": 512,
            "num_epochs": 250
        }
    },
    
    "shallow_wide": {
        "description": "Shallow but wide (2 layers, 512 hidden)",
        "config": "default",
        "params": {
            "num_layers": 2,
            "hidden_dim": 512,
            "num_epochs": 200
        }
    },
    
    # Attention variations
    "more_heads": {
        "description": "More attention heads (8 heads)",
        "config": "default",
        "params": {
            "num_heads": 8,
            "num_epochs": 200
        }
    },
    
    "attention_pooling": {
        "description": "Use attention-based pooling",
        "config": "default",
        "params": {
            "pool_type": "attention",
            "num_epochs": 200
        }
    },
    
    # Regularization experiments
    "high_dropout": {
        "description": "Higher dropout (0.3) for regularization",
        "config": "default",
        "params": {
            "dropout": 0.3,
            "num_epochs": 250
        }
    },
    
    "low_dropout": {
        "description": "Lower dropout (0.05) for more capacity",
        "config": "default",
        "params": {
            "dropout": 0.05,
            "num_epochs": 200
        }
    },
    
    "strong_regularization": {
        "description": "Strong regularization (high dropout + weight decay)",
        "config": "default",
        "params": {
            "dropout": 0.3,
            "weight_decay": 1e-4,
            "num_epochs": 250
        }
    },
    
    # Learning rate experiments
    "high_lr": {
        "description": "Higher learning rate (5e-3)",
        "config": "default",
        "params": {
            "learning_rate": 5e-3,
            "num_epochs": 150
        }
    },
    
    "low_lr": {
        "description": "Lower learning rate (1e-4)",
        "config": "default",
        "params": {
            "learning_rate": 1e-4,
            "num_epochs": 300
        }
    },
    
    # Batch size experiments
    "large_batch": {
        "description": "Larger batch size (64)",
        "config": "default",
        "params": {
            "batch_size": 64,
            "num_epochs": 200
        }
    },
    
    "small_batch": {
        "description": "Smaller batch size (16)",
        "config": "default",
        "params": {
            "batch_size": 16,
            "learning_rate": 5e-4,  # Adjust LR for small batch
            "num_epochs": 250
        }
    },
    
    # Quick tests (for debugging)
    "quick_test": {
        "description": "Quick test run (5 epochs)",
        "config": "small",
        "params": {
            "num_epochs": 5,
            "batch_size": 16
        }
    },
    
    "medium_test": {
        "description": "Medium test run (20 epochs)",
        "config": "default",
        "params": {
            "num_epochs": 20
        }
    },
}


def create_experiment_config(experiment_name: str, custom_params: dict = None) -> Config:
    """
    Create configuration for a named experiment.
    
    Args:
        experiment_name: Name of experiment from EXPERIMENTS dict
        custom_params: Additional parameters to override
        
    Returns:
        Config object
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available: {list(EXPERIMENTS.keys())}")
    
    exp_config = EXPERIMENTS[experiment_name]
    
    # Load base config
    if exp_config["config"] == "small":
        config = get_small_config()
    elif exp_config["config"] == "large":
        config = get_large_config()
    else:
        config = get_default_config()
    
    # Apply experiment parameters
    params = exp_config["params"].copy()
    if custom_params:
        params.update(custom_params)
    
    # Update config
    for key, value in params.items():
        if hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
    
    return config


def setup_experiment_directory(experiment_name: str, output_dir: Path) -> Path:
    """
    Create organized directory structure for experiment.
    
    Args:
        experiment_name: Name of experiment
        output_dir: Base output directory
        
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = output_dir / f"{experiment_name}_{timestamp}"
    
    # Create subdirectories
    (exp_dir / "tensorboard_logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def save_experiment_info(exp_dir: Path, experiment_name: str, config: Config):
    """Save experiment metadata."""
    info = {
        "experiment_name": experiment_name,
        "description": EXPERIMENTS.get(experiment_name, {}).get("description", ""),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model.__dict__,
            "training": config.training.__dict__,
            "data": {k: str(v) if isinstance(v, Path) else v 
                    for k, v in config.data.__dict__.items()}
        }
    }
    
    with open(exp_dir / "experiment_info.json", 'w') as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Run training experiments with predefined or custom configurations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiment',
        type=str,
        default='baseline',
        help=f'Experiment name. Options: {", ".join(EXPERIMENTS.keys())}'
    )
    
    parser.add_argument(
        '--list-experiments',
        action='store_true',
        help='List all available experiments and exit'
    )
    
    # Data arguments (required)
    parser.add_argument(
        '--data-dir',
        type=str,
        required='--list-experiments' not in sys.argv,
        help='Directory with batch_*.pt files (default: data/processed)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        help='Path to raw log files directory (external location, for loading from results.csv)'
    )
    
    parser.add_argument(
        '--labels-file',
        type=str,
        help='Path to labels JSON file (default: data-dir/labels.json)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help='Base directory for all experiments'
    )
    
    # Custom parameter overrides
    parser.add_argument('--hidden-dim', type=int, help='Override hidden dimension')
    parser.add_argument('--num-layers', type=int, help='Override number of layers')
    parser.add_argument('--num-heads', type=int, help='Override number of heads')
    parser.add_argument('--dropout', type=float, help='Override dropout rate')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--num-epochs', type=int, help='Override number of epochs')
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--seed', type=int, help='Override random seed')
    
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list_experiments:
        print("\nAvailable Experiments:")
        print("=" * 80)
        for name, config in EXPERIMENTS.items():
            print(f"\n{name}:")
            print(f"  Description: {config['description']}")
            print(f"  Base config: {config['config']}")
            print(f"  Parameters: {config['params']}")
        print("\n" + "=" * 80)
        return 0
    
    # Validate arguments
    if args.results_csv and not args.log_dir:
        parser.error("--log-dir is required when using --results-csv")
    
    # Create custom parameters dict
    custom_params = {}
    if args.hidden_dim:
        custom_params['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        custom_params['num_layers'] = args.num_layers
    if args.num_heads:
        custom_params['num_heads'] = args.num_heads
    if args.dropout is not None:
        custom_params['dropout'] = args.dropout
    if args.learning_rate:
        custom_params['learning_rate'] = args.learning_rate
    if args.batch_size:
        custom_params['batch_size'] = args.batch_size
    if args.num_epochs:
        custom_params['num_epochs'] = args.num_epochs
    if args.device:
        custom_params['device'] = args.device
    if args.seed:
        custom_params['seed'] = args.seed
    
    # Create configuration
    print(f"\nSetting up experiment: {args.experiment}")
    config = create_experiment_config(args.experiment, custom_params)
    
    # Setup paths
    config.data.data_dir = Path(args.data_dir)
    if args.log_dir:
        config.data.log_dir = Path(args.log_dir)
    if args.labels_file:
        config.data.labels_file = Path(args.labels_file)
    
    # Setup experiment directory
    output_dir = Path(args.output_dir)
    exp_dir = setup_experiment_directory(args.experiment, output_dir)
    
    config.experiment.log_dir = exp_dir / "tensorboard_logs"
    config.experiment.checkpoint_dir = exp_dir / "checkpoints"
    config.experiment.results_dir = exp_dir / "results"
    config.experiment.experiment_name = args.experiment
    
    # Save experiment info
    save_experiment_info(exp_dir, args.experiment, config)
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Description: {EXPERIMENTS[args.experiment]['description']}")
    
    # Run training
    try:
        history, test_metrics = train_model(config)
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"EXPERIMENT '{args.experiment}' COMPLETED")
        print("=" * 80)
        print(f"Results saved to: {exp_dir}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test AUROC: {test_metrics['auroc']:.4f}")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Experiment '{args.experiment}' failed")
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

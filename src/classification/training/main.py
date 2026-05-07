"""
Main training script for multi-task GNN root-cause analysis.

Usage (multi-task with manifest):
    python main.py --manifest-file /path/to/manifest.csv --data-dir data/processed

Usage (legacy binary mode with labels.json):
    python main.py --data-dir data/processed --labels-file data/processed/labels.json
"""

import torch
import numpy as np
import random
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

# Ensure pattern-analysis/ is on sys.path so 'src.*' imports resolve
_ROOT = Path(__file__).resolve().parent.parent.parent  # pattern-analysis/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    Config, DataConfig, ModelConfig, TrainingConfig,
    get_default_config, get_small_config,
    get_medium_config, get_large_config, get_xlarge_config,
    validate_config_for_data,
)
from model import TraceGNN, print_model_summary, count_parameters
from dataset import (
    TraceDataset, split_dataset, create_dataloaders,
    compute_pos_weight, print_dataset_statistics,
)
from trainer import Trainer
from visualization import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log'),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(config: Config):
    """Main training function."""
    logger.info("=" * 80)
    logger.info("GNN ROOT-CAUSE ANALYSIS — MULTI-TASK TRAINING")
    logger.info("=" * 80)

    set_seed(config.data.seed)

    # ── per-run output directories ────────────────────────────────────────────
    # We don't know num_samples yet (need to load dataset first), so we stamp
    # with timestamp now and patch in the sample count after loading.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Temporary run name — will be updated once we know num_samples
    _tmp_run_name = f"run_{timestamp}"
    base_out = config.data.output_dir / _tmp_run_name
    config.experiment.checkpoint_dir = base_out / 'checkpoints'
    config.experiment.log_dir        = base_out / 'tensorboard'
    config.experiment.results_dir    = base_out / 'results'
    config.experiment.run_name       = _tmp_run_name

    config.setup_directories()

    # ── auto-detect input dimension ──────────────────────────────────────────
    if config.model.input_dim is None:
        logger.info("Auto-detecting input dimension from preprocessing...")
        config.model.input_dim = config.detect_input_dim()

    logger.info(f"\nConfiguration:")
    logger.info(f"  input_dim:           {config.model.input_dim}")
    logger.info(f"  hidden_dim:          {config.model.hidden_dim}")
    logger.info(f"  num_layers:          {config.model.num_layers}")
    logger.info(f"  num_heads:           {config.model.num_heads}")
    logger.info(f"  num_agents:          {config.model.num_agents}")
    logger.info(f"  num_failure_classes: {config.model.num_failure_classes}")
    logger.info(f"  lambda_failure:      {config.training.lambda_failure}")
    logger.info(f"  learning_rate:       {config.training.learning_rate}")
    logger.info(f"  batch_size:          {config.data.batch_size}")
    logger.info(f"  num_epochs:          {config.training.num_epochs}")

    # ── load dataset ─────────────────────────────────────────────────────────
    logger.info("\nLoading dataset...")
    dataset = TraceDataset(
        data_dir=config.data.data_dir,
        labels_file=config.data.labels_file,
        log_dir=config.data.log_dir,
        manifest_file=config.data.manifest_file,
    )
    print_dataset_statistics(dataset)

    # ── rename output dirs now that we know num_samples ──────────────────────
    n_samples = len(dataset)
    run_name = f"run_{timestamp}_n{n_samples}"
    new_base = config.data.output_dir / run_name
    # Rename the temp directory we already created
    if base_out.exists():
        base_out.rename(new_base)
    config.experiment.checkpoint_dir = new_base / 'checkpoints'
    config.experiment.log_dir        = new_base / 'tensorboard'
    config.experiment.results_dir    = new_base / 'results'
    config.experiment.run_name       = run_name
    config.setup_directories()
    logger.info(f"Run directory: {new_base}")

    # ── if multi-task, sync num_agents / num_violation_classes from manifest ──
    if dataset.label_mode == 'multi_task':
        n_agents     = len(dataset.agent_mapping)
        n_violations = len(dataset.violation_mapping)
        if (n_agents != config.model.num_agents or
                n_violations != config.model.num_failure_classes):
            logger.info(
                f"Updating model config from manifest: "
                f"num_agents {config.model.num_agents}→{n_agents}, "
                f"num_failure_classes {config.model.num_failure_classes}→{n_violations}"
            )
            config.model.num_agents = n_agents
            config.model.num_failure_classes = n_violations

        # Save label encodings alongside results for later interpretation
        encodings = {
            'agent_mapping':     dataset.agent_mapping,
            'violation_mapping': dataset.violation_mapping,
        }
        enc_path = config.experiment.results_dir / 'label_encodings.json'
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(enc_path, 'w') as f:
            json.dump(encodings, f, indent=2)
        logger.info(f"Saved label encodings to {enc_path}")

    validate_config_for_data(config, len(dataset))

    # ── split ─────────────────────────────────────────────────────────────────
    logger.info("\nSplitting dataset (stratified)...")
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.seed,
    )
    logger.info(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    logger.info("\nInitialising model...")
    model = TraceGNN(
        llm_feature_dim=config.model.input_dim,
        non_llm_feature_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        pool_type=config.model.pool_type,
        use_residual=config.model.use_residual,
        use_layer_norm=config.model.use_layer_norm,
        num_agents=config.model.num_agents,
        num_failure_classes=config.model.num_failure_classes,
    )
    print_model_summary(model)
    logger.info(f"Total parameters: {count_parameters(model):,}")

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=config.training.device,
    )

    # ── train ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80 + "\n")
    history = trainer.train()

    # ── test with best model ──────────────────────────────────────────────────
    best_model_path = config.experiment.checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        logger.info("\nLoading best model for final evaluation...")
        trainer.load_checkpoint(best_model_path)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 80 + "\n")
    test_metrics = trainer.test()

    # ── save loss history as JSON ─────────────────────────────────────────────
    loss_history = {
        'train_loss':         history['train_loss'],
        'train_loss_agent':   history['train_loss_agent'],
        'train_loss_failure': history['train_loss_failure'],
        'val_loss':           history['val_loss'],
        'val_loss_agent':     history['val_loss_agent'],
        'val_loss_failure':   history['val_loss_failure'],
    }
    loss_path = config.experiment.results_dir / 'loss_history.json'
    with open(loss_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    logger.info(f"Saved per-head loss history to {loss_path}")

    # ── save test metrics ─────────────────────────────────────────────────────
    serialisable_metrics = {
        k: (v.tolist() if hasattr(v, 'tolist') else v)
        for k, v in test_metrics.items()
        if not k.startswith('confusion_matrix')
    }
    metrics_path = config.experiment.results_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(serialisable_metrics, f, indent=2)
    logger.info(f"Saved test metrics to {metrics_path}")

    # ── visualisations ────────────────────────────────────────────────────────
    try:
        generate_all_plots(
            history=history,
            test_metrics=test_metrics,
            confusion_matrix=test_metrics.get('confusion_matrix_agent'),
            output_dir=config.experiment.results_dir,
        )
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results:     {config.experiment.results_dir}")
    logger.info(f"Checkpoints: {config.experiment.checkpoint_dir}")

    return test_metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-task GNN for execution trace root-cause analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', type=str,
                        choices=['small', 'medium', 'large', 'xlarge', 'default'],
                        default='default')

    # Data paths
    parser.add_argument('--data-dir',      type=Path, default=Path('data/processed'))
    parser.add_argument('--labels-file',   type=Path, default=None,
                        help='Binary mode: path to labels.json')
    parser.add_argument('--log-dir',       type=Path, default=None,
                        help='Binary mode: path to raw log files with results.csv')
    parser.add_argument('--manifest-file', type=Path, default=None,
                        help='Multi-task mode: path to manifest.csv')

    # Model / training overrides
    parser.add_argument('--num-epochs',       type=int,   default=None)
    parser.add_argument('--batch-size',       type=int,   default=None)
    parser.add_argument('--learning-rate',    type=float, default=None)
    parser.add_argument('--hidden-dim',       type=int,   default=None)
    parser.add_argument('--num-layers',       type=int,   default=None)
    parser.add_argument('--lambda-failure',   type=float, default=None,
                        help='Weight of failure-type loss (default: 1.0)')
    parser.add_argument('--num-agents',       type=int,   default=None,
                        help='Override number of agent classes (auto from manifest)')
    parser.add_argument('--num-failure-classes', type=int, default=None,
                        help='Override number of failure classes (auto from manifest)')

    parser.add_argument('--no-early-stopping', action='store_true')
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--device', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Base config
    presets = {
        'small': get_small_config, 'medium': get_medium_config,
        'large': get_large_config, 'xlarge': get_xlarge_config,
    }
    config = presets.get(args.config, get_default_config)()

    # ── data paths ────────────────────────────────────────────────────────────
    config.data.data_dir = args.data_dir

    if args.manifest_file:
        config.data.manifest_file = args.manifest_file
    elif args.labels_file:
        config.data.labels_file = args.labels_file
    elif args.log_dir:
        config.data.log_dir = args.log_dir
    else:
        # Default: look for manifest or labels.json next to data
        default_manifest = args.data_dir / 'manifest.csv'
        default_labels   = args.data_dir / 'labels.json'
        if default_manifest.exists():
            config.data.manifest_file = default_manifest
            logger.info(f"Auto-detected manifest: {default_manifest}")
        elif default_labels.exists():
            config.data.labels_file = default_labels
            logger.info(f"Auto-detected labels: {default_labels}")

    # ── overrides ─────────────────────────────────────────────────────────────
    if args.num_epochs       is not None: config.training.num_epochs       = args.num_epochs
    if args.batch_size       is not None: config.data.batch_size           = args.batch_size
    if args.learning_rate    is not None: config.training.learning_rate    = args.learning_rate
    if args.hidden_dim       is not None: config.model.hidden_dim          = args.hidden_dim
    if args.num_layers       is not None: config.model.num_layers          = args.num_layers
    if args.lambda_failure   is not None: config.training.lambda_failure   = args.lambda_failure
    if args.num_agents       is not None: config.model.num_agents          = args.num_agents
    if args.num_failure_classes is not None:
        config.model.num_failure_classes = args.num_failure_classes
    if args.no_early_stopping:            config.training.early_stopping   = False
    if args.device           is not None: config.training.device           = args.device
    config.data.seed = args.seed

    try:
        test_metrics = train_model(config)
        logger.info("\n✓ Training completed successfully!")
        logger.info(f"Agent   F1: {test_metrics['agent_f1']:.4f}")
        logger.info(f"Failure F1: {test_metrics['failure_f1']:.4f}")
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

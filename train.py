#!/usr/bin/env python3
"""
Train the TraceGNN classifier on VIOLA.

This is the top-level training entry point (Papers with Code item 2).
Internal training logic lives in src/classification/training/.

Usage:
    python train.py                                    # full run, default config
    python train.py --config configs/default.yaml
    python train.py --quick                            # 5 epochs, small model, CPU-feasible
    python train.py --output-dir results/my_run

Prerequisites:
    python scripts/download_dataset.py    # download dataset
    # then optionally preprocess raw logs (skip if using pre-cached graphs):
    python train.py --preprocess --log-dir data/raw_logs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "classification"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml",
                   help="Path to YAML config (default: configs/default.yaml)")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed",
                   help="Directory of preprocessed .pt graph files")
    p.add_argument("--manifest-file", type=Path, default=ROOT / "data" / "manifest.csv",
                   help="Manifest CSV with labels (downloaded with the dataset)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gnn",
                   help="Directory for checkpoints and logs")
    p.add_argument("--preprocess", action="store_true",
                   help="Run preprocessing before training (requires --log-dir)")
    p.add_argument("--log-dir", type=Path, default=ROOT / "data" / "raw_logs",
                   help="Raw OTel log directory (for --preprocess)")
    p.add_argument("--quick", action="store_true",
                   help="5 epochs, small model — sanity check on CPU (~5 min)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Optional: preprocessing                                              #
    # ------------------------------------------------------------------ #
    if args.preprocess:
        if not args.log_dir.exists():
            print(f"ERROR: --log-dir {args.log_dir} does not exist.")
            sys.exit(1)
        print(f"Preprocessing logs from {args.log_dir} ...")
        from src.classification.preprocessing.main import run_preprocessing
        run_preprocessing(
            log_dir=args.log_dir,
            output_dir=args.data_dir,
            config_name="small" if args.quick else "medium",
        )

    if not args.data_dir.exists() or not list(args.data_dir.glob("*.pt")):
        print(f"ERROR: No preprocessed graphs found in {args.data_dir}.")
        print("  Run: python train.py --preprocess  OR  python scripts/download_dataset.py")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Config                                                               #
    # ------------------------------------------------------------------ #
    import torch
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from src.classification.training.config import get_default_config, get_small_config
    config = get_small_config() if args.quick else get_default_config()
    if args.quick:
        config.training.num_epochs = 5

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    from src.classification.training.dataset import TraceDataset, split_dataset, create_dataloaders
    dataset = TraceDataset(data_dir=str(args.data_dir), manifest_file=str(args.manifest_file))
    train_ds, val_ds, test_ds = split_dataset(dataset, seed=args.seed)
    train_loader, val_loader, _ = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=config.training.batch_size
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    from src.classification.training.model import TraceGNN, print_model_summary
    sample = dataset[0]
    llm_dim = sample.x_llm.shape[1] if hasattr(sample, "x_llm") else sample.x.shape[1]
    non_llm_dim = sample.x.shape[1]

    model = TraceGNN(
        llm_input_dim=llm_dim,
        non_llm_input_dim=non_llm_dim,
        hidden_dim=config.model.hidden_dim,
        num_agents=config.model.num_agents,
        num_failure_classes=config.model.num_failure_classes,
    )
    print_model_summary(model)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    from src.classification.training.trainer import Trainer
    trainer = Trainer(model=model, config=config.training,
                      output_dir=str(args.output_dir))
    trainer.fit(train_loader, val_loader)

    checkpoint_path = args.output_dir / "best_model.pt"
    print(f"\nTraining complete. Best checkpoint: {checkpoint_path}")
    print(f"Evaluate with: python eval.py --model {checkpoint_path}")


if __name__ == "__main__":
    main()

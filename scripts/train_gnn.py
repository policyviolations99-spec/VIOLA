#!/usr/bin/env python3
"""
Train the TraceGNN classifier on preprocessed graph data.

This is a thin wrapper around src/classification/training/main.py that sets
up paths and passes through all arguments.

Prerequisite: run preprocess step first:
    python scripts/train_gnn.py --preprocess --data-dir data/processed

Usage:
    # Full training run (GPU recommended)
    python scripts/train_gnn.py --data-dir data/processed

    # Quick sanity check (~5 epochs, small model)
    python scripts/train_gnn.py --data-dir data/processed --quick

    # Specify output directory
    python scripts/train_gnn.py --data-dir data/processed --output-dir results/gnn
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "classification"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed",
                        help="Directory containing preprocessed .pt graph files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gnn",
                        help="Directory for checkpoints and training logs")
    parser.add_argument("--manifest-file", type=Path, default=ROOT / "data" / "manifest.csv",
                        help="Path to manifest CSV with labels")
    parser.add_argument("--config", default="medium",
                        choices=["small", "medium", "large", "xlarge"],
                        help="Model/training configuration preset")
    parser.add_argument("--quick", action="store_true",
                        help="Run 5 epochs with a small config (sanity check)")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run preprocessing step before training")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "data" / "raw_logs",
                        help="Directory of raw OTel log files (for --preprocess)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.preprocess:
        print("Running preprocessing...")
        from src.classification.preprocessing.main import run_preprocessing
        run_preprocessing(
            log_dir=args.log_dir,
            output_dir=args.data_dir,
            config_name="medium",
        )

    if args.quick:
        args.config = "small"

    print(f"Training TraceGNN (config={args.config}, output={args.output_dir})")

    from src.classification.training.config import get_default_config, get_small_config
    from src.classification.training.model import TraceGNN
    from src.classification.training.dataset import TraceDataset, create_dataloaders, split_dataset
    from src.classification.training.trainer import Trainer

    config = get_small_config() if args.quick else get_default_config()
    if args.quick:
        config.training.num_epochs = 5

    dataset = TraceDataset(
        data_dir=str(args.data_dir),
        manifest_file=str(args.manifest_file),
    )
    train_ds, val_ds, test_ds = split_dataset(dataset)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=config.training.batch_size
    )

    sample = dataset[0]
    model = TraceGNN(
        llm_input_dim=sample.x_llm.shape[1] if hasattr(sample, "x_llm") else sample.x.shape[1],
        non_llm_input_dim=sample.x.shape[1],
        hidden_dim=config.model.hidden_dim,
        num_agents=config.model.num_agents,
        num_failure_classes=config.model.num_failure_classes,
    )

    trainer = Trainer(
        model=model,
        config=config.training,
        output_dir=str(args.output_dir),
    )
    trainer.fit(train_loader, val_loader)

    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(test_loader, split="test")
    print(f"Test results: {metrics}")


if __name__ == "__main__":
    main()

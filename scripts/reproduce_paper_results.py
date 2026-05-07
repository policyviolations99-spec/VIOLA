#!/usr/bin/env python3
"""
Reproduce the main classification results from the paper.

Fast path (uses pre-trained checkpoints, <30 min):
    python scripts/reproduce_paper_results.py

From-scratch path (trains all models, ~2 hr on single GPU):
    python scripts/reproduce_paper_results.py --from-scratch

Quick sanity check (CPU, ~5 min):
    python scripts/reproduce_paper_results.py --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "classification"))


def print_table(results: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Agent F1':>10} {'Violation F1':>14} {'Avg F1':>8}")
    print("-" * 70)
    for model_name, metrics in sorted(results.items()):
        agent_f1 = metrics.get("agent_f1", 0.0)
        viol_f1 = metrics.get("violation_f1", 0.0)
        avg_f1 = (agent_f1 + viol_f1) / 2
        print(f"  {model_name:<28} {agent_f1:>10.3f} {viol_f1:>14.3f} {avg_f1:>8.3f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Dataset root directory (default: data/)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results",
                        help="Directory for outputs (default: results/)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train all models from scratch instead of using pre-trained checkpoints")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 5 epochs, small model, skip some baselines")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (use cached data/processed/ graphs)")
    parser.add_argument("--skip-gnn", action="store_true",
                        help="Skip GNN (only run baselines)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baselines (only run GNN)")
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    manifest_file = args.data_dir / "manifest.csv"
    log_dir = args.data_dir / "raw_logs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Check prerequisites ---
    parquet_files = list(args.data_dir.glob("*.parquet"))
    if not parquet_files:
        print("ERROR: Dataset not found. Run first:")
        print("  python scripts/download_dataset.py")
        sys.exit(1)

    # --- Step 1: Preprocessing ---
    if not args.skip_preprocess:
        if not log_dir.exists():
            print("WARNING: raw_logs/ directory not found. Skipping preprocessing.")
            print("  Set --skip-preprocess if you have pre-cached processed graphs.")
        else:
            print(f"Step 1: Preprocessing OTel logs from {log_dir} ...")
            from src.classification.preprocessing.main import run_preprocessing
            run_preprocessing(
                log_dir=log_dir,
                output_dir=processed_dir,
                config_name="small" if args.quick else "medium",
            )
    else:
        print("Step 1: Skipping preprocessing (using cached graphs)")

    if not processed_dir.exists() or not list(processed_dir.glob("*.pt")):
        print(f"ERROR: No preprocessed graphs found in {processed_dir}")
        print("  Run without --skip-preprocess, or download raw logs first.")
        sys.exit(1)

    # --- Setup dataset ---
    from src.classification.training.dataset import TraceDataset, create_dataloaders, split_dataset
    dataset = TraceDataset(data_dir=str(processed_dir), manifest_file=str(manifest_file))
    train_ds, val_ds, test_ds = split_dataset(dataset)

    batch_size = 16 if args.quick else 32
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )

    sample = dataset[0]
    results = {}

    # --- Step 2: TraceGNN ---
    if not args.skip_gnn:
        print("\nStep 2: Training TraceGNN ...")
        from src.classification.training.config import get_small_config, get_default_config
        from src.classification.training.model import TraceGNN
        from src.classification.training.trainer import Trainer

        config = get_small_config() if args.quick else get_default_config()
        if args.quick:
            config.training.num_epochs = 5

        model = TraceGNN(
            llm_input_dim=sample.x_llm.shape[1] if hasattr(sample, "x_llm") else sample.x.shape[1],
            non_llm_input_dim=sample.x.shape[1],
            hidden_dim=config.model.hidden_dim,
            num_agents=config.model.num_agents,
            num_failure_classes=config.model.num_failure_classes,
        )
        trainer = Trainer(model=model, config=config.training,
                          output_dir=str(args.output_dir / "gnn"))
        trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(test_loader, split="test")
        results["TraceGNN"] = metrics
        print(f"  TraceGNN test metrics: {metrics}")

    # --- Step 3: Baselines ---
    if not args.skip_baselines:
        print("\nStep 3: Training baselines ...")
        from src.baselines.graph_baselines import (
            LinearBaseline, MLPBaseline, GCNBaseline, GraphSAGEBaseline
        )

        baselines = [
            ("Linear (mean-pool)", LinearBaseline),
            ("MLP (mean-pool)", MLPBaseline),
            ("GCN", GCNBaseline),
            ("GraphSAGE", GraphSAGEBaseline),
        ]
        if args.quick:
            baselines = baselines[:2]

        for name, ModelClass in baselines:
            print(f"  Training {name} ...")
            config = get_small_config() if args.quick else get_default_config()
            if args.quick:
                config.training.num_epochs = 5
            model = ModelClass(
                in_dim=sample.x.shape[1],
                num_agents=config.model.num_agents,
                num_failure_classes=config.model.num_failure_classes,
            )
            trainer = Trainer(model=model, config=config.training,
                              output_dir=str(args.output_dir / name.replace(" ", "_").replace("(", "").replace(")", "")))
            trainer.fit(train_loader, val_loader)
            metrics = trainer.evaluate(test_loader, split="test")
            results[name] = metrics
            print(f"    {name}: {metrics}")

    # --- Print results table ---
    print_table(results)

    # Save JSON
    results_path = args.output_dir / "paper_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate a trained TraceGNN (or baseline) checkpoint on a dataset split.

This is the top-level evaluation entry point (Papers with Code item 3).

Usage:
    # Evaluate the released GNN checkpoint on the test split
    python eval.py --model checkpoints/gnn_main.pt --split test

    # Evaluate a locally trained checkpoint
    python eval.py --model results/gnn/best_model.pt --split test

    # Evaluate a baseline checkpoint
    python eval.py --model checkpoints/gcn_baseline.pt --split test

    # Evaluate on validation split
    python eval.py --model checkpoints/gnn_main.pt --split val

Outputs a JSON metrics dict with agent_f1, violation_f1, agent_accuracy,
violation_accuracy, and per-class breakdowns.
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
    p.add_argument("--model", type=Path, required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--split", choices=["train", "val", "test"], default="test",
                   help="Dataset split to evaluate on (default: test)")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed",
                   help="Directory of preprocessed .pt graph files")
    p.add_argument("--manifest-file", type=Path, default=ROOT / "data" / "manifest.csv",
                   help="Manifest CSV with labels")
    p.add_argument("--output", type=Path, default=None,
                   help="Save metrics JSON to this path (optional)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.model.exists():
        print(f"ERROR: checkpoint not found: {args.model}")
        print("  Run: python scripts/download_pretrained.py  to download released checkpoints.")
        sys.exit(1)

    if not args.data_dir.exists() or not list(args.data_dir.glob("*.pt")):
        print(f"ERROR: No preprocessed graphs found in {args.data_dir}.")
        print("  Run: python scripts/download_dataset.py")
        sys.exit(1)

    import torch
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------ #
    # Load checkpoint                                                      #
    # ------------------------------------------------------------------ #
    print(f"Loading checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu")

    model_config = checkpoint.get("model_config", {})
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    from src.classification.training.model import TraceGNN
    from src.classification.training.config import get_default_config

    config = get_default_config()
    model = TraceGNN(
        llm_input_dim=model_config.get("llm_input_dim", config.model.llm_input_dim),
        non_llm_input_dim=model_config.get("non_llm_input_dim", config.model.non_llm_input_dim),
        hidden_dim=model_config.get("hidden_dim", config.model.hidden_dim),
        num_agents=model_config.get("num_agents", config.model.num_agents),
        num_failure_classes=model_config.get("num_failure_classes", config.model.num_failure_classes),
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded.")

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    from src.classification.training.dataset import TraceDataset, split_dataset, create_dataloaders
    dataset = TraceDataset(data_dir=str(args.data_dir), manifest_file=str(args.manifest_file))
    train_ds, val_ds, test_ds = split_dataset(dataset, seed=args.seed)

    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    eval_ds = split_map[args.split]
    _, _, loader = create_dataloaders(train_ds, val_ds, test_ds,
                                      batch_size=32, shuffle=False)
    if args.split == "val":
        from torch_geometric.loader import DataLoader
        loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    elif args.split == "train":
        from torch_geometric.loader import DataLoader
        loader = DataLoader(train_ds, batch_size=32, shuffle=False)

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    from src.classification.training.trainer import Trainer
    trainer = Trainer(model=model, config=config.training,
                      output_dir=str(ROOT / "results" / "eval"))
    metrics = trainer.evaluate(loader, split=args.split)

    print(f"\n{'=' * 50}")
    print(f"Results on {args.split} split:")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
    print(f"{'=' * 50}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")

    return metrics


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train baseline classifiers for comparison against TraceGNN.

Available baselines:
  linear   — logistic regression on mean-pooled node features
  mlp      — 2-layer MLP on mean-pooled features (no graph structure)
  gcn      — Graph Convolutional Network (Kipf & Welling 2017)
  sage     — GraphSAGE (inductive neighborhood aggregation)

Usage:
    python scripts/train_baselines.py --baseline all
    python scripts/train_baselines.py --baseline gcn
    python scripts/train_baselines.py --baseline gcn --quick
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "classification"))

BASELINE_MAP = {
    "linear": ("LinearBaseline",  "linear_baseline.pt"),
    "mlp":    ("MLPBaseline",     "mlp_baseline.pt"),
    "gcn":    ("GCNBaseline",     "gcn_baseline.pt"),
    "sage":   ("GraphSAGEBaseline", "sage_baseline.pt"),
}


def train_baseline(name: str, cls_name: str, filename: str, args) -> dict:
    import torch
    from src.baselines.graph_baselines import LinearBaseline, MLPBaseline, GCNBaseline, GraphSAGEBaseline
    from src.classification.training.config import get_default_config, get_small_config
    from src.classification.training.dataset import TraceDataset, split_dataset, create_dataloaders
    from src.classification.training.trainer import Trainer

    cls_map = {
        "LinearBaseline": LinearBaseline,
        "MLPBaseline": MLPBaseline,
        "GCNBaseline": GCNBaseline,
        "GraphSAGEBaseline": GraphSAGEBaseline,
    }

    config = get_small_config() if args.quick else get_default_config()
    if args.quick:
        config.training.num_epochs = 5

    dataset = TraceDataset(data_dir=str(args.data_dir), manifest_file=str(args.manifest_file))
    train_ds, val_ds, test_ds = split_dataset(dataset, seed=args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=config.training.batch_size
    )

    sample = dataset[0]
    model = cls_map[cls_name](
        in_dim=sample.x.shape[1],
        num_agents=config.model.num_agents,
        num_failure_classes=config.model.num_failure_classes,
    )

    out_dir = args.output_dir / name
    trainer = Trainer(model=model, config=config.training, output_dir=str(out_dir))
    trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(test_loader, split="test")

    ckpt_path = out_dir / filename
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "class": cls_name,
            "in_dim": sample.x.shape[1],
            "num_agents": config.model.num_agents,
            "num_failure_classes": config.model.num_failure_classes,
        },
    }, ckpt_path)
    print(f"  [{name}] checkpoint saved: {ckpt_path}")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", choices=list(BASELINE_MAP.keys()) + ["all"], default="all")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--manifest-file", type=Path, default=ROOT / "data" / "manifest.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "baselines")
    p.add_argument("--quick", action="store_true", help="5 epochs, small model")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    baselines = list(BASELINE_MAP.items()) if args.baseline == "all" else [(args.baseline, BASELINE_MAP[args.baseline])]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for name, (cls_name, filename) in baselines:
        print(f"\nTraining {name} baseline ...")
        metrics = train_baseline(name, cls_name, filename, args)
        all_metrics[name] = metrics
        print(f"  [{name}] test metrics: {metrics}")

    print("\n" + "=" * 50)
    print("Baseline results summary:")
    for name, metrics in all_metrics.items():
        agent_f1 = metrics.get("agent_f1", 0.0)
        viol_f1  = metrics.get("violation_f1", 0.0)
        print(f"  {name:<10}  agent_f1={agent_f1:.3f}  violation_f1={viol_f1:.3f}")

    out_path = args.output_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

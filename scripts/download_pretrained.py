#!/usr/bin/env python3
"""
Download pre-trained model checkpoints for VIOLA.

Downloads to ./checkpoints/ and verifies SHA-256 integrity.

Usage:
    python scripts/download_pretrained.py              # all checkpoints
    python scripts/download_pretrained.py --model gnn  # GNN only

After downloading, evaluate with:
    python eval.py --model checkpoints/gnn_main.pt --split test
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"

CHECKPOINTS = {
    "gnn": {
        "filename": "gnn_main.pt",
        "url": "https://huggingface.co/policy-violation-benchmark/viola-models/resolve/main/gnn_main.pt",
        "sha256": "88ddb16d6e5aabf101fe099e5447a1cdfebd0da26680957ca3ca6e15139891fb",
        "description": "TraceGNN main checkpoint — reproduces headline paper results (seed 45)",
    },
    "gcn": {
        "filename": "gcn_baseline.pt",
        "url": "https://huggingface.co/policy-violation-benchmark/viola-models/resolve/main/gcn_baseline.pt",
        "sha256": "284617a9071a0bb82215a1a7edec0907879a61b46dc7665c61d20944276c4545",
        "description": "GCN baseline checkpoint (seed 42)",
    },
    "sage": {
        "filename": "sage_baseline.pt",
        "url": "https://huggingface.co/policy-violation-benchmark/viola-models/resolve/main/sage_baseline.pt",
        "sha256": "9bc7bee833656701af550414183c09d72a96a5b736567b6e309e50bb49aea099",
        "description": "GraphSAGE baseline checkpoint (seed 42)",
    },
    "mlp": {
        "filename": "mlp_baseline.pt",
        "url": "https://huggingface.co/policy-violation-benchmark/viola-models/resolve/main/mlp_baseline.pt",
        "sha256": "fad850f1e4dbe8966efa3c8d8549adcc3388619ad3e5d2346bcb70964453a92d",
        "description": "MLP (mean-pool) baseline checkpoint (seed 42)",
    },
    "linear": {
        "filename": "linear_baseline.pt",
        "url": "https://huggingface.co/policy-violation-benchmark/viola-models/resolve/main/linear_baseline.pt",
        "sha256": "6b71b9193ee0962aebe96678788d7016c7e04468c6628cfbad901dc569426beb",
        "description": "Linear (mean-pool) baseline checkpoint (seed 42)",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(name: str, info: dict, dest_dir: Path, verify: bool = True) -> Path:
    dest = dest_dir / info["filename"]

    if "TODO" in info["url"]:
        print(f"  [{name}] SKIPPED — URL not yet configured (see TODO in scripts/download_pretrained.py)")
        return dest

    if dest.exists():
        print(f"  [{name}] Already exists: {dest}")
    else:
        print(f"  [{name}] Downloading {info['description']} ...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(info["url"], dest)
        print(f"  [{name}] Saved to {dest}")

    if verify and "TODO" not in info["sha256"]:
        actual = sha256_file(dest)
        if actual != info["sha256"]:
            print(f"  [{name}] SHA-256 MISMATCH — expected {info['sha256']}, got {actual}")
            dest.unlink()
            sys.exit(1)
        print(f"  [{name}] Checksum OK")

    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(CHECKPOINTS.keys()) + ["all"], default="all",
                   help="Which checkpoint to download (default: all)")
    p.add_argument("--output-dir", type=Path, default=CHECKPOINTS_DIR,
                   help=f"Destination directory (default: {CHECKPOINTS_DIR})")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip SHA-256 verification")
    args = p.parse_args()

    models = list(CHECKPOINTS.keys()) if args.model == "all" else [args.model]

    print(f"Downloading checkpoints to {args.output_dir} ...")
    for name in models:
        download_checkpoint(name, CHECKPOINTS[name], args.output_dir,
                            verify=not args.no_verify)

    print("\nDone. Evaluate with:")
    print("  python eval.py --model checkpoints/gnn_main.pt --split test")


if __name__ == "__main__":
    main()

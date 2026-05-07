#!/usr/bin/env python3
"""
Download the VIOLA dataset from HuggingFace.

Downloads all splits and supplementary files (policies, metadata) to
a local data/ directory and verifies file integrity.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --data-dir /path/to/custom/dir
"""

import argparse
import sys
from pathlib import Path

HF_DATASET_ID = "policy-violation-benchmark/VIOLA"
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def download_dataset(data_dir: Path) -> None:
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: Missing dependencies. Run: pip install datasets huggingface_hub")
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset {HF_DATASET_ID} to {data_dir} ...")

    # Download the main dataset splits
    ds = load_dataset(HF_DATASET_ID, cache_dir=str(data_dir / ".cache"))
    print(f"  Splits: {list(ds.keys())}")
    for split, dataset in ds.items():
        out = data_dir / f"{split}.parquet"
        dataset.to_parquet(str(out))
        print(f"  Saved {split} split ({len(dataset)} rows) -> {out}")

    # Download supplementary files (policies, metadata, raw_logs)
    print("\nDownloading supplementary files (policies, metadata) ...")
    snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(data_dir),
        ignore_patterns=["*.cache", ".git*"],
    )
    print(f"  Supplementary files saved to {data_dir}")

    # Verify expected files exist
    required = [
        data_dir / "train.parquet",
        data_dir / "validation.parquet",
        data_dir / "test.parquet",
        data_dir / "policies" / "original" / "APIPlannerAgent.md",
        data_dir / "metadata" / "violation_taxonomy.json",
    ]
    missing = [f for f in required if not f.exists()]
    if missing:
        print("\nWARNING: Some expected files are missing:")
        for f in missing:
            print(f"  {f}")
    else:
        print("\nAll required files verified successfully.")
        print(f"\nDataset ready at: {data_dir}")
        print("Next step: python scripts/reproduce_paper_results.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory to download the dataset to (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    download_dataset(args.data_dir)


if __name__ == "__main__":
    main()

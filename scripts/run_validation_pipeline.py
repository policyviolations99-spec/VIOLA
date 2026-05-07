#!/usr/bin/env python3
"""
Run the 4-stage validation pipeline on a directory of OTel trace logs.

This is useful if you want to validate your own generated traces, or re-run
validation on the raw logs included in the dataset download.

Prerequisites:
  - JUDGE_MODEL environment variable: model name (e.g., "gpt-4.1")
  - JUDGE_BASE_URL environment variable: OpenAI-compatible API base URL
  - JUDGE_API_KEY environment variable: API key

Usage:
    # Validate a directory of generated traces
    python scripts/run_validation_pipeline.py \\
        --run-dir /path/to/run_dir \\
        --output-dir results/validation_output

    # Validate with a cap on number of traces (for testing)
    python scripts/run_validation_pipeline.py \\
        --run-dir /path/to/run_dir \\
        --max-traces 10

    # Use a custom policies directory
    POLICIES_DIR=/path/to/policies/original \\
        python scripts/run_validation_pipeline.py --run-dir /path/to/run_dir
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.pipeline import run_validation
from src.validation.judge_runner import verify_and_select_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the run output directory (must contain manifest.json + logs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save per-trace validation results",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of traces to validate (for testing)",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include traces where the agent run failed (skipped by default)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override path to manifest.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.run_dir.exists():
        print(f"ERROR: run_dir does not exist: {args.run_dir}")
        sys.exit(1)

    print("VIOLA — Validation Pipeline")
    print("=" * 60)
    print(f"Run dir:     {args.run_dir}")
    print(f"Output dir:  {args.output_dir or '(none)'}")
    print(f"Max traces:  {args.max_traces or 'unlimited'}")
    print()

    print("Initializing judge model...")
    model = verify_and_select_model()
    print(f"Using model: {model}\n")

    results = run_validation(
        run_dir=str(args.run_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        skip_failed=not args.include_failed,
        max_traces=args.max_traces,
    )

    # Summary
    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(results)} traces")
    print(f"  Accepted: {len(accepted)} ({100 * len(accepted) / max(len(results), 1):.1f}%)")
    print(f"  Rejected: {len(rejected)} ({100 * len(rejected) / max(len(results), 1):.1f}%)")

    if accepted:
        hard = sum(1 for r in accepted if r.final_label == "hard")
        soft = sum(1 for r in accepted if r.final_label == "soft")
        print(f"  Labels: {hard} hard, {soft} soft")

    if args.output_dir:
        summary_path = args.output_dir / "validation_summary.json"
        summary = {
            "total": len(results),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "results": [r.to_dict() for r in results],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    main()

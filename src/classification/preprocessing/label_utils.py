"""
Label loading utilities for trace datasets.

Loads labels from results.csv (binary success/failure) or manifest.csv
(multi-task: agent identification + failure type classification).
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import re

logger = logging.getLogger(__name__)


def extract_trace_number(filename: str) -> int:
    """
    Extract trace number from log filename.
    
    Args:
        filename: e.g., "trace01.log", "trace123.log", "trace1.log"

    Returns:
        Trace number as integer (1, 123, etc.)
    """
    match = re.search(r'trace(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract trace number from: {filename}")


def normalize_trace_filename(filename: str) -> str:
    """
    Normalize trace filename to match training expectations.

    For trace numbers < 10: Remove zero-padding (trace01.log -> trace1.log)
    For trace numbers >= 10: Keep as-is (trace10.log, trace123.log)

    Args:
        filename: Original filename (e.g., "trace01.log", "trace1.log", "trace10.log")

    Returns:
        Normalized filename (e.g., "trace1.log", "trace1.log", "trace10.log")
    """
    trace_num = extract_trace_number(filename)

    if trace_num < 10:
        # Remove zero-padding for single digits
        return f"trace{trace_num}.log"
    else:
        # Keep as-is for double digits and above
        return filename


def load_labels_from_results_csv(
    log_dir: Path,
    batch_size: int = 100
) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Load labels from results.csv (in same directory as log files).

    The results.csv has one row per log file, ordered by filename:
    - Row 0: Headers
    - Row 1: First log file (alphabetically sorted)
    - Row 2: Second log file (alphabetically sorted)
    - etc.

    Log filenames are normalized to match training expectations:
    - trace1.log, trace2.log, ..., trace9.log (no zero-padding)
    - trace10.log, trace11.log, ... (keep as-is)

    Args:
        log_dir: Path to directory with .log files and results.csv (external/raw data)
        batch_size: Number of traces per batch (default 100, matching preprocessing)

    Returns:
        Tuple of (log_filename_to_label dict, full_dataframe)
    """
    log_dir = Path(log_dir)
    results_csv_path = log_dir / "results.csv"

    if not results_csv_path.exists():
        raise FileNotFoundError(
            f"results.csv not found in {log_dir}. "
            f"Expected location: {results_csv_path}"
        )

    logger.info(f"Loading labels from {results_csv_path}")

    # Load results.csv
    df = pd.read_csv(results_csv_path)
    logger.info(f"Loaded {len(df)} rows from results.csv")

    # Verify required columns exist
    required_cols = ['score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in results.csv: {missing_cols}")

    # Log available columns
    logger.info(f"Available columns: {list(df.columns)}")

    # Get sorted list of log files (using standard string sort)
    log_files = sorted(log_dir.glob("*.log"))
    logger.info(f"Found {len(log_files)} log files in {log_dir}")

    # Show first few filenames to verify sort order
    logger.info(f"First 10 log files (as found): {[f.name for f in log_files[:10]]}")

    # Verify alignment
    if len(df) != len(log_files):
        logger.warning(
            f"Mismatch: results.csv has {len(df)} rows but found {len(log_files)} log files. "
            f"This may cause label misalignment!"
        )

    # Create mapping: normalized_log_filename -> label
    log_filename_to_label = {}

    # Track metadata for verification
    trace_metadata = {}

    for idx, row in df.iterrows():
        # Get corresponding log file
        if idx >= len(log_files):
            logger.warning(f"Row {idx}: No corresponding log file (only {len(log_files)} files)")
            break

        log_file = log_files[idx]
        original_filename = log_file.name  # e.g., "trace01.log" or "trace1.log"

        # Normalize filename to match training expectations
        normalized_filename = normalize_trace_filename(original_filename)

        # Get label (score column is binary success/failure)
        label = int(row['score'])

        if label not in [0, 1]:
            logger.warning(f"Row {idx}: Invalid label value {label}, expected 0 or 1. Skipping.")
            continue

        # Check for duplicates after normalization
        if normalized_filename in log_filename_to_label:
            logger.warning(
                f"Row {idx}: Duplicate normalized filename '{normalized_filename}' "
                f"(original: '{original_filename}'). Overwriting previous label."
            )

        log_filename_to_label[normalized_filename] = label

        # Store metadata for verification
        metadata = {
            'csv_row': idx,
            'original_filename': original_filename,
            'normalized_filename': normalized_filename,
            'label': label
        }

        # Add optional metadata if columns exist
        if 'task_id' in df.columns:
            metadata['task_id'] = row['task_id']
        if 'exception' in df.columns:
            metadata['exception'] = bool(row['exception'])

        trace_metadata[normalized_filename] = metadata

    logger.info(f"Created labels for {len(log_filename_to_label)} traces (after normalization)")

    # Print label distribution
    label_counts = pd.Series(list(log_filename_to_label.values())).value_counts().sort_index()
    logger.info(f"\nLabel distribution:")
    logger.info(f"  Failure (0): {label_counts.get(0, 0)}")
    logger.info(f"  Success (1): {label_counts.get(1, 0)}")

    if 'exception' in df.columns:
        exception_rate = df['exception'].mean()
        logger.info(f"\nException rate: {exception_rate:.2%}")

        # Cross-check: traces with exceptions should typically be failures
        df_with_labels = df.copy()
        df_with_labels['predicted_failure'] = df_with_labels['exception'].astype(int)
        agreement = (df_with_labels['score'] == (1 - df_with_labels['predicted_failure'])).mean()
        logger.info(f"Exception/failure agreement: {agreement:.2%}")

    # Print sample mappings for verification
    logger.info("\nSample log_filename -> label mappings (first 10 normalized):")
    for i, normalized_filename in enumerate(sorted(log_filename_to_label.keys(), key=lambda x: extract_trace_number(x))[:10]):
        meta = trace_metadata[normalized_filename]
        logger.info(
            f"  {meta['original_filename']} -> {normalized_filename}: "
            f"label={meta['label']}, csv_row={meta['csv_row']}"
            + (f", task_id={meta.get('task_id', 'N/A')}" if 'task_id' in meta else "")
        )

    return log_filename_to_label, df


def verify_label_alignment(
    log_filename_to_label: Dict[str, int],
    log_dir: Path,
    batch_size: int = 100
) -> bool:
    """
    Verify that label alignment is correct by checking log filenames.

    Args:
        log_filename_to_label: Mapping from normalized log_filename to label
        log_dir: Directory with log files
        batch_size: Batch size used in preprocessing

    Returns:
        True if alignment looks correct
    """
    logger.info("\nVerifying label alignment...")

    log_files = sorted(log_dir.glob("*.log"))

    # Check a few specific alignments
    test_indices = [0, 1, 10, 50] if len(log_files) > 50 else [0, 1, min(10, len(log_files)-1)]

    all_correct = True

    for file_idx in test_indices:
        if file_idx >= len(log_files):
            continue

        log_file = log_files[file_idx]
        original_filename = log_file.name
        normalized_filename = normalize_trace_filename(original_filename)

        # Check if label exists
        has_label = normalized_filename in log_filename_to_label

        status = "✓" if has_label else "✗"

        logger.info(
            f"{status} File {file_idx}: {original_filename} -> {normalized_filename} "
            f"(has_label: {has_label})"
        )

        if not has_label:
            all_correct = False
            logger.warning(
                f"  WARNING: No label found for {normalized_filename}"
            )

    if all_correct:
        logger.info("\n✓ Label alignment verification passed!")
    else:
        logger.warning("\n✗ Label alignment issues detected. Please verify manually.")

    return all_correct


def save_labels_json(
    log_filename_to_label: Dict[str, int],
    output_path: Path
):
    """
    Save labels to JSON file (should be in data/processed directory).
    Keys are sorted by trace number (natural ordering).

    Args:
        log_filename_to_label: Mapping from normalized log_filename to label
        output_path: Path to save JSON file (typically data/processed/labels.json)
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort keys by trace number (natural ordering)
    sorted_labels = {k: log_filename_to_label[k]
                    for k in sorted(log_filename_to_label.keys(),
                                   key=lambda x: extract_trace_number(x))}

    with open(output_path, 'w') as f:
        json.dump(sorted_labels, f, indent=2)

    logger.info(f"Saved {len(sorted_labels)} labels to {output_path}")
    logger.info(f"Keys are normalized (trace1.log, trace2.log, ..., trace10.log, trace11.log, ...)")


def load_labels_json(labels_path: Path) -> Dict[str, int]:
    """
    Load labels from JSON file.

    Args:
        labels_path: Path to labels JSON file

    Returns:
        Dictionary mapping trace_id to label
    """
    import json

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    # Ensure all values are integers
    labels = {str(k): int(v) for k, v in labels.items()}

    logger.info(f"Loaded {len(labels)} labels from {labels_path}")

    return labels


def load_labels_from_manifest_csv(
    manifest_path: Path,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, int]]:
    """
    Load multi-task labels from a manifest.csv file.

    Each row in the manifest corresponds to one log file, identified by the
    basename of the 'log_path' column (e.g. 'task_abc_PlanControllerAgent_F01.log').

    Supports two manifest schemas:
      - Legacy:  'failure_id' column  (e.g. F01, F02, ...)
      - Current: 'violation_id' column (e.g. V1, V2, ...)

    Rows with empty 'log_path' (crashed/failed runs) are skipped automatically.

    Returns integer-encoded labels for two tasks:
      - agent:     which agent had the failure injected (0, 1, ...)
      - violation: which violation type was injected (0, 1, ..., N-1)

    Args:
        manifest_path: Path to manifest.csv

    Returns:
        Tuple of:
          - labels:            {log_filename: {'agent': int, 'violation': int}}
          - agent_mapping:     {agent_name: int}  (e.g. {'PlanControllerAgent': 0, ...})
          - violation_mapping: {violation_id: int} (e.g. {'V1': 0, 'V2': 1, ...})
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found at: {manifest_path}")

    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(df)} rows from {manifest_path}")

    # Detect violation column: prefer 'violation_id', fall back to 'failure_id'
    if 'violation_id' in df.columns:
        violation_col = 'violation_id'
    elif 'failure_id' in df.columns:
        violation_col = 'failure_id'
    else:
        raise ValueError(
            "manifest.csv missing violation column: expected 'violation_id' or 'failure_id'"
        )
    logger.info(f"Using '{violation_col}' as the violation/failure column")

    required_cols = ['log_path', 'agent', violation_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"manifest.csv missing required columns: {missing}")

    # Drop rows with missing or empty log_path (crashed/failed runs have no log)
    total_before = len(df)
    df = df[df['log_path'].notna() & (df['log_path'].str.strip() != '')]
    skipped = total_before - len(df)
    if skipped:
        logger.info(
            f"Skipped {skipped} rows with empty log_path "
            f"(crashed/failed runs without agent invocation)"
        )

    # Build deterministic label encodings (sorted for reproducibility)
    agent_names = sorted(df['agent'].unique().tolist())
    violation_ids = sorted(df[violation_col].unique().tolist())

    agent_mapping: Dict[str, int] = {name: idx for idx, name in enumerate(agent_names)}
    violation_mapping: Dict[str, int] = {vid: idx for idx, vid in enumerate(violation_ids)}

    logger.info(f"Agent mapping: {agent_mapping}")
    logger.info(f"Violation mapping: {violation_mapping}")

    labels: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        log_filename = Path(row['log_path']).name  # e.g. 'task_abc_APIPlannerAgent_V1_r1.log'
        labels[log_filename] = {
            'agent': agent_mapping[row['agent']],
            'violation': violation_mapping[row[violation_col]],
        }

    logger.info(f"Created multi-task labels for {len(labels)} traces")
    logger.info(f"  Agents ({len(agent_mapping)}): {agent_names}")
    logger.info(f"  Violation types ({len(violation_mapping)}): {violation_ids}")

    return labels, agent_mapping, violation_mapping


if __name__ == "__main__":
    """
    Standalone script to generate labels.json from results.csv.
    
    Usage:
        python label_utils.py <log_dir> [output_path]
        
    Example:
        python label_utils.py /external/path/to/logs
        # Creates: data/processed/labels.json (default)
        
        python label_utils.py /external/path/to/logs data/processed/labels.json
        # Creates: data/processed/labels.json (explicit)
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python label_utils.py <log_dir> [output_path]")
        print("\nArguments:")
        print("  log_dir      - Directory with .log files and results.csv (raw data location)")
        print("  output_path  - Where to save labels.json (default: data/processed/labels.json)")
        print("\nExample:")
        print("  python label_utils.py /external/data/logs")
        print("  # Creates: data/processed/labels.json")
        sys.exit(1)

    log_dir = Path(sys.argv[1])

    # Default: save to data/processed/labels.json
    if len(sys.argv) > 2:
        output_json = Path(sys.argv[2])
    else:
        output_json = Path("data/processed/labels.json")

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    results_csv = log_dir / "results.csv"
    if not results_csv.exists():
        print(f"Error: results.csv not found at: {results_csv}")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load labels
    log_filename_to_label, df = load_labels_from_results_csv(log_dir)

    # Verify alignment
    verify_label_alignment(log_filename_to_label, log_dir)

    # Save to JSON
    save_labels_json(log_filename_to_label, output_json)

    print(f"\n{'='*80}")
    print("✓ Labels created successfully!")
    print(f"{'='*80}")
    print(f"Location: {output_json}")
    print(f"Total traces: {len(log_filename_to_label)}")
    print(f"  Success: {sum(1 for v in log_filename_to_label.values() if v == 1)}")
    print(f"  Failure: {sum(1 for v in log_filename_to_label.values() if v == 0)}")
    print(f"\nKeys are normalized to match training expectations:")
    print(f"  - trace1.log, trace2.log, ..., trace9.log (no zero-padding)")
    print(f"  - trace10.log, trace11.log, ... (as-is for >= 10)")
    print(f"\nNext step:")
    print(f"  python main.py --data-dir {output_json.parent} --labels-file {output_json}")
    print(f"{'='*80}")
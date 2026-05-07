"""
Main entry point for execution trace preprocessing.

This script processes execution log files and converts them into graph representations
suitable for GNN training. Supports batch processing with checkpointing.

Usage:
    # List available models
    python -m preprocessing.main list-models

    # Run preprocessing with medium model (default)
    python -m preprocessing.main run --log_dir /path/to/logs --output_dir ../../data/processed

    # Run with xlarge model
    python -m preprocessing.main run --config xlarge --log_dir /path/to/logs

    # Test single file
    python -m preprocessing.main test --test_file /path/to/logs/trace_001.log
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Ensure pattern-analysis/ is on sys.path so 'src.*' imports resolve
# and src/preprocessing/ is on sys.path so bare 'utils.*' imports resolve.
_ROOT = Path(__file__).resolve().parent.parent.parent  # pattern-analysis/
_PREPROCESSING = Path(__file__).resolve().parent       # src/preprocessing/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_PREPROCESSING) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING))

from src.utils.tasks_metrics_fetcher import initialize_fetcher, get_tasks_metrics
from src.preprocessing.config import (
    PreprocessingConfig,
    get_default_config,
    get_small_config,
    get_large_config,
    get_xlarge_config
)

logger = logging.getLogger(__name__)


def list_models():
    """List all available embedding models with details."""
    from src.preprocessing.utils.embeddings import print_available_models
    print_available_models()


def test_single_file(file_path: Path) -> None:
    """
    Test preprocessing on a single log file.

    This function validates that:
    1. The log file can be loaded
    2. Tasks can be extracted
    3. LLM calls can be identified

    Args:
        file_path: Path to a single .log file
    """
    logger.info(f"=" * 80)
    logger.info(f"Testing with single file: {file_path}")
    logger.info(f"=" * 80)

    if not file_path.exists():
        logger.error(f"Test file not found: {file_path}")
        return

    try:
        # Initialize fetcher
        initialize_fetcher()
        logger.info("✓ Fetcher initialized")

        # Load tasks
        tasks_dict, metrics_list = get_tasks_metrics(str(file_path.resolve()))
        logger.info(f"✓ Loaded {len(tasks_dict)} tasks from file")

        # Analyze tasks
        llm_tasks = [
            task_id for task_id, task in tasks_dict.items()
            if 'llm_call' in task.tags
        ]
        non_llm_tasks = [
            task_id for task_id, task in tasks_dict.items()
            if 'llm_call' not in task.tags
        ]

        logger.info(f"  - LLM call tasks: {len(llm_tasks)}")
        logger.info(f"  - Non-LLM tasks: {len(non_llm_tasks)}")

        # Show sample task IDs
        if tasks_dict:
            sample_ids = list(tasks_dict.keys())[:3]
            logger.info(f"  - Sample task IDs: {sample_ids}")

            # Show first task structure
            first_task = tasks_dict[sample_ids[0]]
            logger.info(f"  - First task keys: {list(first_task.keys())[:10]}")

        logger.info("✓ Single file test completed successfully")

    except Exception as e:
        logger.error(f"✗ Error during single file test: {e}", exc_info=True)
        raise


def get_log_files(log_dir: Path) -> list[Path]:
    """
    Get all .log files from directory.

    Args:
        log_dir: Directory containing log files

    Returns:
        List of Path objects for .log files
    """
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    log_files = sorted(log_dir.glob("*.log"))

    if not log_files:
        raise ValueError(f"No .log files found in {log_dir}")

    return log_files


def run_preprocessing(config: PreprocessingConfig) -> None:
    """
    Run the full preprocessing pipeline.

    Args:
        config: Preprocessing configuration
    """
    logger.info("=" * 80)
    logger.info("Starting Preprocessing Pipeline")
    logger.info("=" * 80)

    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  - Embedding model: {config.embedding.model_name}")
    logger.info(f"  - Embedding dimension: {config.embedding.dimension}")
    logger.info(f"  - Total LLM feature dim: {config.get_total_llm_dim()}")
    logger.info(f"  - Batch size: {config.batching.batch_size}")
    logger.info(f"  - Start from batch: {config.batching.start_from_batch}")
    logger.info(f"  - Log directory: {config.log_dir}")
    logger.info(f"  - Output directory: {config.output_dir}")

    # Setup directories
    config.setup_directories()
    logger.info("✓ Directories setup complete")

    # Get log files
    log_files = get_log_files(config.log_dir)
    logger.info(f"✓ Found {len(log_files)} log files")

    # Run the pipeline
    from src.preprocessing.pipeline import preprocess_traces
    preprocess_traces(log_files, config)

    logger.info("=" * 80)
    logger.info("Preprocessing Complete")
    logger.info("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess execution traces for GNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # =========================================================================
    # LIST-MODELS command
    # =========================================================================
    subparsers.add_parser(
        'list-models',
        help='List all available embedding models with specifications'
    )

    # =========================================================================
    # RUN command
    # =========================================================================
    run_parser = subparsers.add_parser(
        'run',
        help='Run the full preprocessing pipeline'
    )

    run_parser.add_argument(
        '--config',
        type=str,
        choices=['small', 'medium', 'large', 'xlarge'],
        default='medium',
        help='Embedding model size (default: medium). Use list-models to see details.'
    )

    run_parser.add_argument(
        '--log_dir',
        type=Path,
        required=True,
        help='Directory containing .log files'
    )

    run_parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Output directory for processed data (default: ../../data/processed)'
    )

    run_parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )

    run_parser.add_argument(
        '--start_from_batch',
        type=int,
        default=0,
        help='Resume from specific batch number (default: 0)'
    )

    # =========================================================================
    # TEST command
    # =========================================================================
    test_parser = subparsers.add_parser(
        'test',
        help='Test preprocessing on a single log file'
    )

    test_parser.add_argument(
        '--test_file',
        type=Path,
        required=True,
        help='Path to a single .log file for testing'
    )

    return parser.parse_args()


def setup_logging(level="INFO", log_dir=None):
    """
    Configure logging for the entire pipeline.
    
    Args:
        level: Logging level
        log_dir: Optional directory for log files (creates with timestamp)
    """
    handlers = [logging.StreamHandler()]  # Console

    # Add file handler if log_dir specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"preprocessing_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file))
        print(f"Logging to: {log_file}")

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )

def main():
    """Main entry point."""
    # Setup basic logging first (file logging added later if needed)
    setup_logging(level="INFO")
    
    args = parse_args()

    # Handle commands
    if args.command == 'list-models':
        list_models()
        return

    elif args.command == 'test':
        test_single_file(args.test_file)
        return

    elif args.command == 'run':
        # Get base configuration based on model size
        if args.config == 'small':
            config = get_small_config()
        elif args.config == 'large':
            config = get_large_config()
        elif args.config == 'xlarge':
            config = get_xlarge_config()
        else:
            config = get_default_config()

        # Override with command line arguments
        config.log_dir = args.log_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        else:
            # Default: project_root/data/processed (two levels up from src/preprocessing)
            config.output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
            config.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"

        config.batching.batch_size = args.batch_size
        config.batching.start_from_batch = args.start_from_batch

        # Add file logging to output directory
        setup_logging(level="INFO", log_dir=config.output_dir / "logs")

        run_preprocessing(config)

    else:
        # No command specified, show help
        parser = argparse.ArgumentParser()
        parser.print_help()


if __name__ == "__main__":
    main()

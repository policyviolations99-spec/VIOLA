"""
Main preprocessing pipeline with batch processing.

This orchestrates:
1. Pass 1: Collect system prompts and build cache
2. Pass 2: Process traces in batches and convert to PyG Data
3. Save preprocessing metadata for training validation
"""

import logging
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np

from src.utils.tasks_metrics_fetcher import initialize_fetcher, get_tasks_metrics
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.node_extraction import extract_nodes_and_edges, group_nodes_by_type
from src.preprocessing.llm_encoding.prompt_cache import build_prompt_cache, PromptCache
from src.preprocessing.llm_encoding.llm_feature_builder import build_llm_features_batch
from src.preprocessing.non_llm_encoding.non_llm_features import build_non_llm_features_batch
from src.preprocessing.common_features.base_features import compute_common_features_batch
from src.preprocessing.output.pyg_converter import convert_to_pyg_data, save_pyg_data, print_data_statistics
from src.preprocessing.utils.embeddings import load_embedding_model
from task_filtering import filter_tasks

logger = logging.getLogger(__name__)


def load_trace(log_file: Path) -> Tuple[dict, list]:
    """
    Load a single trace from log file.

    Args:
        log_file: Path to .log file

    Returns:
        Tuple of (tasks_dict, metrics_list)
    """
    tasks_dict, metrics_list = get_tasks_metrics(str(log_file.resolve()))
    return tasks_dict, metrics_list


def chunks(lst: List, n: int):
    """
    Yield successive n-sized chunks from list.

    Args:
        lst: List to chunk
        n: Chunk size

    Yields:
        Chunks of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_single_trace(
        tasks_dict: dict,
        prompt_cache: PromptCache,
        embedding_model,
        config: PreprocessingConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Process a single trace into features.

    Args:
        tasks_dict: Dictionary of tasks
        prompt_cache: PromptCache instance
        embedding_model: EmbeddingModel instance
        config: Configuration

    Returns:
        Tuple of (llm_features, non_llm_features, common_features, edges, node_types)
    """
    # Extract nodes and edges
    nodes, edges = extract_nodes_and_edges(tasks_dict)

    # Group nodes by type
    grouped_nodes = group_nodes_by_type(nodes)
    llm_nodes = grouped_nodes.get('llm_call', [])
    non_llm_nodes = grouped_nodes.get('non_llm', [])

    # Build LLM features
    if llm_nodes:
        llm_task_data = [node.task_data for node in llm_nodes]
        llm_features = build_llm_features_batch(
            llm_task_data,
            prompt_cache,
            embedding_model,
            batch_size=32
        )
    else:
        llm_features = np.zeros((0, config.get_total_llm_dim()), dtype=np.float32)

    # Build non-LLM features (zero-padded)
    if non_llm_nodes:
        non_llm_features = build_non_llm_features_batch(
            non_llm_nodes,
            target_dim=config.get_total_llm_dim()
        )
    else:
        non_llm_features = np.zeros((0, config.get_total_llm_dim()), dtype=np.float32)

    # Build common features for all nodes
    common_features = compute_common_features_batch(nodes)

    # Extract node types (binary: 0=llm, 1=non-llm)
    node_types = np.array([0 if node.node_type == 'llm_call' else 1 for node in nodes], dtype=np.int64)

    return llm_features, non_llm_features, common_features, edges, node_types


def save_preprocessing_metadata(config: PreprocessingConfig, num_traces: int):
    """
    Save preprocessing metadata for training validation.
    
    Args:
        config: Preprocessing configuration
        num_traces: Number of traces processed
    """
    metadata = {
        'embedding_model': config.embedding.model_name,
        'embedding_size': config.embedding.size,
        'embedding_dimension': config.embedding.dimension,
        'llm_feature_dim': config.get_total_llm_dim(),
        'common_feature_dim': 40,  # Updated to 40 with binary node types
        'total_feature_dim': config.get_total_llm_dim() + 40,
        'num_traces': num_traces,
        'batch_size': config.batching.batch_size,
        'node_types': {
            '0': 'llm_call',
            '1': 'non_llm'
        }
    }

    metadata_file = config.output_dir / 'preprocessing_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved preprocessing metadata to {metadata_file}")
    logger.info(f"  - Total feature dimension: {metadata['total_feature_dim']}")
    logger.info(f"  - LLM feature dimension: {metadata['llm_feature_dim']}")
    logger.info(f"  - Common feature dimension: {metadata['common_feature_dim']}")


def preprocess_traces(log_files: List[Path], config: PreprocessingConfig) -> None:
    """
    Main preprocessing pipeline with batching.

    Args:
        log_files: List of .log file paths
        config: Preprocessing configuration
    """
    logger.info("=" * 80)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    # Initialize fetcher
    initialize_fetcher()
    logger.info("✓ Task fetcher initialized")

    # =========================================================================
    # PASS 1: Build Prompt Cache
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PASS 1: Building Prompt Cache")
    logger.info("=" * 80)

    # Try to load existing cache
    prompt_cache = PromptCache(config.cache_dir, config.features.role_signature_dim)
    cache_loaded = prompt_cache.load()

    if not cache_loaded or len(prompt_cache) == 0:
        logger.info("No existing cache found, building new cache...")

        # Collect all nodes from all traces
        all_nodes = []
        logger.info(f"Scanning {len(log_files)} log files for system prompts...")

        for i, log_file in enumerate(log_files):
            if (i + 1) % 10 == 0:
                logger.info(f"  Scanned {i + 1}/{len(log_files)} files...")

            try:
                tasks_dict, _ = load_trace(log_file)
                nodes, _ = extract_nodes_and_edges(tasks_dict)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.error(f"Error loading {log_file}: {e}")
                continue

        logger.info(f"✓ Collected {len(all_nodes)} nodes total")

        # Build cache
        prompt_cache = build_prompt_cache(
            all_nodes,
            config.cache_dir,
            config.features.role_signature_dim
        )
    else:
        logger.info("✓ Loaded existing prompt cache")

    # =========================================================================
    # PASS 2: Process Traces in Batches
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PASS 2: Processing Traces")
    logger.info("=" * 80)

    # Load embedding model
    logger.info(f"Loading embedding model: {config.embedding.model_name}")
    embedding_model = load_embedding_model(config.embedding.model_name)

    # Process in batches
    total_batches = (len(log_files) + config.batching.batch_size - 1) // config.batching.batch_size
    logger.info(f"Processing {len(log_files)} traces in {total_batches} batches")
    logger.info(f"Batch size: {config.batching.batch_size}")
    logger.info(f"Starting from batch: {config.batching.start_from_batch}")

    total_traces_processed = 0

    for batch_idx, batch_files in enumerate(chunks(log_files, config.batching.batch_size)):
        # Skip if before start batch
        if batch_idx < config.batching.start_from_batch:
            logger.info(f"Skipping batch {batch_idx} (before start_from_batch)")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing Batch {batch_idx}/{total_batches - 1} ({len(batch_files)} traces)")
        logger.info(f"{'=' * 60}")

        batch_data_objects = []

        for i, log_file in enumerate(batch_files):
            try:
                logger.info(f"  [{i + 1}/{len(batch_files)}] Processing {log_file.name}...")

                # Load trace
                tasks_dict, _ = load_trace(log_file)
                if not tasks_dict or len(tasks_dict) <= 1:
                    logger.warning(f"    ⚠ Skipping {log_file.name} - empty trace (0 tasks)")
                    continue
                if config.apply_task_filtering:  # Add this config option
                    tasks_dict = filter_tasks(tasks_dict)
                # Process trace
                llm_features, non_llm_features, common_features, edges, node_types = process_single_trace(
                    tasks_dict,
                    prompt_cache,
                    embedding_model,
                    config
                )

                # Convert to PyG Data
                data = convert_to_pyg_data(
                    llm_features,
                    non_llm_features,
                    common_features,
                    edges,
                    node_types
                )

                # Store original log filename for label lookup
                data.log_filename = log_file.name

                batch_data_objects.append(data)

                logger.info(f"    ✓ {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")

            except Exception as e:
                logger.error(f"    ✗ Error processing {log_file.name}: {e}", exc_info=True)
                continue

        # Save batch
        if batch_data_objects:
            batch_file = config.output_dir / f"batch_{batch_idx}.pt"

            # Save as list of Data objects
            import torch
            torch.save(batch_data_objects, batch_file)

            logger.info(f"\n✓ Saved batch {batch_idx} to {batch_file}")
            logger.info(f"  - {len(batch_data_objects)} traces processed")

            total_traces_processed += len(batch_data_objects)

            # Print statistics for first trace in batch
            # if batch_data_objects:
            #     logger.info("\nSample trace statistics:")
            #     print_data_statistics(batch_data_objects[0])
        else:
            logger.warning(f"Batch {batch_idx} produced no valid traces!")

    # =========================================================================
    # SAVE METADATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SAVING METADATA")
    logger.info("=" * 80)

    save_preprocessing_metadata(config, total_traces_processed)
    # =========================================================================
    # STATISTICS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING STATISTICS")
    logger.info("=" * 80)
    total_files = len(log_files)
    discarded = total_files - total_traces_processed
    logger.info(f"Total log files found: {total_files}")
    logger.info(f"Successfully processed: {total_traces_processed}")
    logger.info(f"Discarded (empty/errors): {discarded}")
    if total_files > 0:
        logger.info(f"Success rate: {100 * total_traces_processed / total_files:.1f}%")

    # =========================================================================
    # DONE
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Batch files saved as: batch_*.pt")
    logger.info(f"Total traces processed: {total_traces_processed}")
    logger.info(f"Preprocessing metadata: preprocessing_metadata.json")

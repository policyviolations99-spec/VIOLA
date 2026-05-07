"""
Configuration management for execution trace preprocessing.

This module defines all configuration parameters for the preprocessing pipeline,
including embedding model selection, feature dimensions, and I/O paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal



@dataclass
class EmbeddingConfig:
    """Configuration for sentence embedding models."""

    size: Literal['small', 'medium', 'large', 'xlarge'] = 'medium'
    dimension: int = field(init=False)  # Will be set in __post_init__
    model_name: str = field(init=False)  # Will be set in __post_init__

    def __post_init__(self):
        """Set model name and dimension based on size."""
        # Import here to avoid circular dependency
        from utils.embeddings import get_model_info

        model_info = get_model_info(self.size)
        self.model_name = model_info.name
        self.dimension = model_info.dimension


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Dimension allocations
    role_signature_dim: int = 32

    # Feature toggles
    include_token_features: bool = True
    include_validity_features: bool = True

    # Validity feature count
    num_validity_features: int = 5  # is_well_formed, output_length, has_error_stems, format_matches, is_truncated

    # Token feature count
    num_token_features: int = 3  # input_tokens, output_tokens, token_ratio

    # Structural feature count (must match STRUCTURAL_DIM in structural_features.py)
    num_structural_features: int = 12

    def get_llm_feature_dim(self, embedding_dim: int) -> int:
        """Calculate total LLM feature dimension."""
        total = (
            self.role_signature_dim +    # 32
            embedding_dim +              # 384/768 (task)
            embedding_dim +              # 384/768 (output)
            self.num_token_features +    # 3
            self.num_validity_features + # 5
            self.num_structural_features # 12
        )
        return total


@dataclass
class BatchingConfig:
    """Configuration for batch processing."""

    batch_size: int = 100
    start_from_batch: int = 0  # Set to N to resume from batch N
    clear_output_on_start: bool = True  # If True and start_from_batch==0, delete existing output


@dataclass
class PreprocessingConfig:
    """Main configuration for preprocessing pipeline."""

    # Sub-configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)

    # I/O paths
    log_dir: Path = Path('./data/logs')
    output_dir: Path = Path('../../data/processed')

    # Prompt cache settings
    cache_dir: Path = Path('../../data/cache')

    apply_task_filtering:bool = True

    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        self.log_dir = Path(self.log_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)

    def setup_directories(self):
        """Create necessary directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Handle output directory based on batching config
        if self.batching.start_from_batch == 0 and self.batching.clear_output_on_start:
            # Starting fresh - clear output directory
            import shutil
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_total_llm_dim(self) -> int:
        """Get total dimension for LLM node features."""
        return self.features.get_llm_feature_dim(self.embedding.dimension)




# Example usage and defaults
def get_default_config() -> PreprocessingConfig:
    """Get default preprocessing configuration (medium model)."""
    return PreprocessingConfig()


def get_small_config() -> PreprocessingConfig:
    """Get configuration for small embedding model (fastest, for testing)."""
    return PreprocessingConfig(
        embedding=EmbeddingConfig(size='small')
    )


def get_large_config() -> PreprocessingConfig:
    """Get configuration for large embedding model (better quality)."""
    return PreprocessingConfig(
        embedding=EmbeddingConfig(size='large')
    )


def get_xlarge_config() -> PreprocessingConfig:
    """Get configuration for xlarge embedding model (best quality, needs 5000+ traces)."""
    return PreprocessingConfig(
        embedding=EmbeddingConfig(size='xlarge')
    )
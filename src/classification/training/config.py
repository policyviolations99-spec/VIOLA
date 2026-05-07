"""
Training configuration for GNN trace analysis.

This module defines all hyperparameters and settings for the training pipeline.
Follows best practices for reproducibility and experiment tracking.

Updated to support flexible input dimensions based on preprocessing model size.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import torch
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """GNN model architecture configuration."""
    
    # Input dimensions (auto-detected from preprocessing or specified)
    input_dim: Optional[int] = None  # Will be auto-detected if None
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4  # For GAT attention
    dropout: float = 0.15
    
    # Aggregation
    pool_type: str = 'mean'  # 'mean', 'max', 'add', or 'attention'
    
    # Normalization
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    
    # Skip connections
    use_residual: bool = True
    
    # Multi-task output dimensions (set from manifest / auto-detected)
    num_agents: int = 2          # number of distinct agents in training data
    num_failure_classes: int = 5 # number of distinct failure types

    def __post_init__(self):
        """Validate configuration."""
        if self.input_dim is not None:
            assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.pool_type in ['mean', 'max', 'add', 'attention']
        assert self.num_agents >= 2, "num_agents must be >= 2"
        assert self.num_failure_classes >= 2, "num_failure_classes must be >= 2"


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    # Paths
    data_dir: Path = Path("data/processed")  # Preprocessed batch_*.pt files
    output_dir: Path = Path("outputs")
    
    # Labels (multiple options)
    labels_file: Optional[Path] = None    # Path to labels.json (binary mode)
    log_dir: Optional[Path] = None        # Path to raw log files (binary mode, for results.csv)
    manifest_file: Optional[Path] = None  # Path to manifest.csv (multi-task mode)
    
    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data augmentation (future)
    use_augmentation: bool = False
    
    # Random seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.labels_file:
            self.labels_file = Path(self.labels_file)
        if self.log_dir:
            self.log_dir = Path(self.log_dir)
        if self.manifest_file:
            self.manifest_file = Path(self.manifest_file)
        
        # Validate splits
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total}"


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Loss function
    loss_fn: str = 'cross_entropy'  # multi-task: always cross_entropy per head

    # Multi-task loss weighting
    lambda_failure: float = 1.0  # weight of failure-type loss relative to agent loss

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'

    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_type: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', 'step'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Training duration
    num_epochs: int = 100

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    early_stopping_min_delta: float = 0.001  # Minimum improvement to count as better

    # Loss weighting (for imbalanced classes)
    use_class_weights: bool = True
    pos_weight: Optional[float] = None  # Auto-computed if None

    # Gradient clipping
    grad_clip_norm: Optional[float] = 1.0

    # Mixed precision training
    use_amp: bool = False

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Logging
    log_every_n_steps: int = 10  # Log training metrics every N steps

    # Checkpointing
    save_every_n_epochs: int = 10  # Save checkpoint every N epochs
    keep_last_n_checkpoints: int = 5  # Keep only the last N checkpoints to save disk space

    def __post_init__(self):
        """Validate configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.patience > 0, "patience must be positive"
        assert self.loss_fn in ['cross_entropy'], \
            "loss_fn must be 'cross_entropy' for multi-task mode"
        assert self.lambda_failure >= 0, "lambda_failure must be non-negative"
        assert self.early_stopping_min_delta >= 0, "early_stopping_min_delta must be non-negative"
        assert self.log_every_n_steps > 0, "log_every_n_steps must be positive"
        assert self.save_every_n_epochs > 0, "save_every_n_epochs must be positive"
        assert self.keep_last_n_checkpoints > 0, "keep_last_n_checkpoints must be positive"

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    # Classification threshold
    classification_threshold: float = 0.5

    # Metrics to compute
    compute_auc: bool = True
    compute_f1: bool = True
    compute_precision_recall: bool = True

    # Evaluation frequency
    eval_every_n_epochs: int = 1

    # Logging
    log_confusion_matrix: bool = True
    save_predictions: bool = False

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.classification_threshold < 1, \
            "classification_threshold must be between 0 and 1"
        assert self.eval_every_n_epochs > 0, \
            "eval_every_n_epochs must be positive"


@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration."""
    
    # Experiment identification
    experiment_name: str = 'trace_gnn'
    run_name: Optional[str] = None  # Auto-generated if None
    
    # Checkpointing
    checkpoint_dir: Path = Path('outputs/checkpoints')
    checkpoint_interval: int = 10  # Save every N epochs
    save_best_only: bool = True
    
    # Logging
    use_tensorboard: bool = True
    log_dir: Path = Path('outputs/tensorboard_logs')
    log_interval: int = 10  # Log every N batches

    # Results
    results_dir: Path = Path('outputs/results')
    save_predictions: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Convert paths and validate."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.results_dir = Path(self.results_dir)


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def setup_directories(self):
        """Create necessary directories."""
        self.data.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment.results_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_input_dim(self) -> int:
        """
        Auto-detect input dimension from preprocessing metadata.
        
        Returns:
            Input dimension
        """
        metadata_file = self.data.data_dir / 'preprocessing_metadata.json'
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                input_dim = metadata['total_feature_dim']
                logger.info(f"✓ Auto-detected input dimension from metadata: {input_dim}")
                logger.info(f"  - Preprocessing model: {metadata['embedding_model']}")
                logger.info(f"  - Embedding size: {metadata['embedding_size']}")
                
                # Validate data requirements
                num_traces = metadata.get('num_traces', 0)
                embedding_size = metadata['embedding_size']
                
                if embedding_size == 'xlarge' and num_traces < 5000:
                    logger.warning(
                        f"⚠ XLarge model used with only {num_traces} traces. "
                        f"Recommended: 5000+ traces. Risk of overfitting!"
                    )
                elif embedding_size == 'medium' and num_traces < 1500:
                    logger.warning(
                        f"⚠ Medium model with only {num_traces} traces. "
                        f"Recommended: 1500+ traces for stable training."
                    )
                
                return input_dim
                
            except Exception as e:
                logger.warning(f"Could not load preprocessing metadata: {e}")
        
        # Fallback to first batch if metadata not found
        logger.warning("Preprocessing metadata not found, detecting from first batch...")
        return self.detect_input_dim_from_batch()
    
    def detect_input_dim_from_batch(self) -> int:
        """
        Detect input dimension from first batch file.
        
        Returns:
            Input dimension
        """
        import torch
        
        batch_files = sorted(self.data.data_dir.glob("batch_*.pt"))
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {self.data.data_dir}")
        
        # Load first batch
        batch = torch.load(batch_files[0], weights_only=False)
        if not batch:
            raise ValueError("First batch is empty")
        
        # Get feature dimension from first graph
        input_dim = batch[0].x.shape[1]
        logger.info(f"✓ Detected input dimension from batch: {input_dim}")
        
        return input_dim


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_default_config() -> Config:
    """Get default configuration (auto-detects input dimension)."""
    return Config()


def get_medium_config() -> Config:
    """
    Configuration for training with medium preprocessing model (850 dims).
    Recommended for 1,500-2,000 traces.
    """
    return Config(
        model=ModelConfig(
            input_dim=850,  # Medium: 808 LLM + 42 common (updated to 40 with binary types)
            hidden_dim=256,
            num_layers=3,
            num_heads=4,
            dropout=0.15
        ),
        training=TrainingConfig(
            learning_rate=3e-4,
            num_epochs=100,
            patience=20
        )
    )


def get_large_config() -> Config:
    """
    Configuration for training with large preprocessing model (850 dims).
    Recommended for 2,000-4,000 traces.
    """
    return Config(
        model=ModelConfig(
            input_dim=850,  # Large: 808 LLM + 42 common (same as medium)
            hidden_dim=384,  # Larger hidden for more capacity
            num_layers=3,
            num_heads=6,
            dropout=0.2
        ),
        training=TrainingConfig(
            learning_rate=2e-4,
            num_epochs=150,
            patience=25
        )
    )


def get_xlarge_config() -> Config:
    """
    Configuration for training with xlarge preprocessing model (1618 dims).
    Recommended for 5,000+ traces ONLY.
    """
    return Config(
        model=ModelConfig(
            input_dim=1618,  # XLarge: 1576 LLM + 42 common (updated to 40)
            hidden_dim=512,  # Much larger for rich inputs
            num_layers=4,
            num_heads=8,
            dropout=0.2
        ),
        training=TrainingConfig(
            learning_rate=1e-4,  # Lower LR for stability
            num_epochs=200,
            patience=30,
            weight_decay=5e-4  # More regularization
        )
    )


def get_small_config() -> Config:
    """
    Small configuration for quick testing/debugging.
    """
    return Config(
        model=ModelConfig(
            input_dim=None,  # Auto-detect
            hidden_dim=128,
            num_layers=2,
            num_heads=2,
            dropout=0.1
        ),
        data=DataConfig(
            batch_size=16,
            num_workers=2
        ),
        training=TrainingConfig(
            learning_rate=5e-4,
            num_epochs=20,
            patience=5
        )
    )


def validate_config_for_data(config: Config, num_traces: int):
    """
    Validate configuration against available data.
    
    Args:
        config: Training configuration
        num_traces: Number of training traces available
    """
    input_dim = config.model.input_dim
    
    if input_dim is None:
        logger.info("Input dimension will be auto-detected from data")
        return
    
    # Rough parameter count estimation
    hidden = config.model.hidden_dim
    layers = config.model.num_layers
    approx_params = input_dim * hidden + (layers - 1) * hidden * hidden
    
    # Rule of thumb: need ~10-20x more samples than parameters
    recommended_traces = approx_params // 100  # Conservative estimate
    
    if num_traces < recommended_traces:
        logger.warning(
            f"⚠ Model has ~{approx_params:,} parameters but only {num_traces} traces. "
            f"Recommended: {recommended_traces:,}+ traces. Risk of overfitting!"
        )
        logger.warning("Consider: reducing hidden_dim or num_layers, or using more data")
    
    # Specific dimension warnings
    if input_dim >= 1600:
        if num_traces < 5000:
            logger.warning(
                f"⚠ XLarge input dimension ({input_dim}) with only {num_traces} traces. "
                f"Strongly recommended: 5000+ traces."
            )
    elif input_dim >= 800:
        if num_traces < 1500:
            logger.warning(
                f"⚠ Medium/Large input dimension ({input_dim}) with only {num_traces} traces. "
                f"Recommended: 1500+ traces."
            )

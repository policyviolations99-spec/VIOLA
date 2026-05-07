"""
Sentence embedding utilities using SentenceTransformers.

This module provides a wrapper around SentenceTransformer models
for consistent text encoding across the preprocessing pipeline.
"""

import logging
import torch
import numpy as np
from typing import List, Union, Dict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

@dataclass
class ModelInfo:
    """Information about an embedding model."""
    name: str  # HuggingFace model name
    dimension: int  # Embedding dimension
    description: str  # Brief description
    size_mb: int  # Approximate model size in MB
    speed: str  # Relative speed: 'fastest', 'fast', 'medium', 'slow'
    recommended_min_traces: int  # Minimum recommended training traces


# Available embedding models with their specifications
EMBEDDING_MODELS: Dict[str, ModelInfo] = {
    'small': ModelInfo(
        name='paraphrase-TinyBERT-L6-v2',
        dimension=768,
        description='Smallest model for fast prototyping and testing',
        size_mb=25,
        speed='fastest',
        recommended_min_traces=500
    ),
    'medium': ModelInfo(
        name='paraphrase-MiniLM-L3-v2',
        dimension=384,
        description='Balanced size and quality. RECOMMENDED for limited data (1,500-4,000 traces)',
        size_mb=60,
        speed='fast',
        recommended_min_traces=1500
    ),
    'large': ModelInfo(
        name='all-MiniLM-L6-v2',
        dimension=384,
        description='Better quality with moderate speed. Good for 2,000+ traces',
        size_mb=90,
        speed='medium',
        recommended_min_traces=2000
    ),
    'xlarge': ModelInfo(
        name='nli-distilroberta-base-v2',
        dimension=768,
        description='NLI-trained for logical relationships. Best quality. Requires 5,000+ traces',
        size_mb=290,
        speed='slow',
        recommended_min_traces=5000
    ),
}


def get_model_info(size: str) -> ModelInfo:
    """
    Get model information by size tag.

    Args:
        size: Model size ('small', 'medium', 'large', 'xlarge')

    Returns:
        ModelInfo object

    Raises:
        ValueError: If size is not recognized
    """
    if size not in EMBEDDING_MODELS:
        available = ', '.join(EMBEDDING_MODELS.keys())
        raise ValueError(f"Unknown model size '{size}'. Available: {available}")

    return EMBEDDING_MODELS[size]


def list_available_models() -> Dict[str, ModelInfo]:
    """
    Get all available models.

    Returns:
        Dictionary of size tag to ModelInfo
    """
    return EMBEDDING_MODELS.copy()


def print_available_models():
    """Print information about all available models."""
    print("\n" + "=" * 100)
    print("AVAILABLE EMBEDDING MODELS")
    print("=" * 100)

    # Header
    print(f"\n{'Size':<10} {'Model Name':<35} {'Dim':<6} {'Size':<8} {'Speed':<10} {'Min Traces':<12}")
    print("-" * 100)

    # Models
    for size, info in EMBEDDING_MODELS.items():
        print(
            f"{size:<10} {info.name:<35} {info.dimension:<6} {info.size_mb}MB{'':<4} {info.speed:<10} {info.recommended_min_traces:<12}")

    print("\n" + "-" * 100)
    print("\nDETAILED DESCRIPTIONS:")
    print("-" * 100)

    for size, info in EMBEDDING_MODELS.items():
        print(f"\n{size.upper()}")
        print(f"  {info.description}")

    print("\n" + "=" * 100)
    print("\nRECOMMENDATIONS:")
    print("  - Start with 'medium' for initial development (1,500-4,000 traces)")
    print("  - Use 'large' when you have 2,000+ high-quality traces")
    print("  - Try 'xlarge' only when you have 5,000+ traces (NLI training helps with logical relationships)")
    print("  - Use 'small' only for quick prototyping and testing")
    print("=" * 100 + "\n")


# =============================================================================
# EMBEDDING MODEL CLASS
# =============================================================================

class EmbeddingModel:
    """
    Wrapper for SentenceTransformer with caching and batch processing.
    """

    def __init__(self, model_name: str, device: str = None):
        """
        Initialize embedding model.

        Args:
            model_name: Name of SentenceTransformer model
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {device}")

        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ Model loaded successfully (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def encode(
            self,
            texts: Union[str, List[str]],
            batch_size: int = 32,
            show_progress: bool = False,
            normalize: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            Embeddings as numpy array of shape (n_texts, dimension)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(texts) if text and text.strip()]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            # All texts empty, return zero vectors
            logger.warning("All texts are empty, returning zero embeddings")
            embeddings = np.zeros((len(texts), self.dimension))
            return embeddings[0] if single else embeddings

        # Encode
        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )

            # If some texts were empty, insert zero vectors
            if len(valid_indices) < len(texts):
                full_embeddings = np.zeros((len(texts), self.dimension))
                full_embeddings[valid_indices] = embeddings
                embeddings = full_embeddings

            return embeddings[0] if single else embeddings

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            # Return zero vectors as fallback
            embeddings = np.zeros((len(texts), self.dimension))
            return embeddings[0] if single else embeddings

    def encode_batch(
            self,
            texts: List[str],
            batch_size: int = 32,
            show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of texts with progress tracking.

        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embeddings as numpy array
        """
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)

    def __repr__(self) -> str:
        return f"EmbeddingModel(model={self.model_name}, dimension={self.dimension}, device={self.device})"


def load_embedding_model(model_name: str, device: str = None) -> EmbeddingModel:
    """
    Load an embedding model.

    Args:
        model_name: Name of SentenceTransformer model
        device: Device to use (None for auto-detect)

    Returns:
        Loaded EmbeddingModel instance
    """
    return EmbeddingModel(model_name, device=device)


class NameEmbedder:
    """
    Lightweight embedder for node names using TF-IDF + SVD.

    Creates compact representations of node names (e.g., "PlanControllerAgent",
    "CodeAgent") to capture functional similarity between nodes.

    Why not use SentenceTransformer?
    - Node names are short identifiers, not sentences
    - TF-IDF works better for short technical terms
    - Much faster and lighter weight
    - Lower dimension (32) is sufficient for names
    """

    def __init__(self, embedding_dim: int = 32):
        """
        Initialize name embedder.

        Args:
            embedding_dim: Dimension for name embeddings (default: 32)
        """
        self.embedding_dim = embedding_dim
        self.vectorizer = None
        self.svd = None
        self.name_to_embedding = {}

        logger.info(f"Initialized NameEmbedder (dim={embedding_dim})")

    def fit(self, names: List[str]) -> None:
        """
        Fit the embedder on a collection of names.

        Args:
            names: List of node names (e.g., ["PlanControllerAgent", "CodeAgent"])
        """
        if not names:
            logger.warning("No names to fit embedder")
            return

        unique_names = list(set(names))
        logger.info(f"Fitting NameEmbedder on {len(unique_names)} unique names...")

        # Build character-level TF-IDF
        # Character-level works better for technical identifiers
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            analyzer='char',
            ngram_range=(2, 4),  # 2-4 character sequences
            lowercase=True
        )

        tfidf_matrix = self.vectorizer.fit_transform(unique_names)
        logger.info(f"  - TF-IDF matrix: {tfidf_matrix.shape}")

        # Apply SVD for dimensionality reduction
        n_components = min(
            self.embedding_dim,
            tfidf_matrix.shape[1],
            tfidf_matrix.shape[0]
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

        embeddings = self.svd.fit_transform(tfidf_matrix)
        logger.info(f"  - SVD reduced to: {embeddings.shape}")

        # Pad to target dimension if needed
        if embeddings.shape[1] < self.embedding_dim:
            padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])

        # Cache embeddings
        for name, embedding in zip(unique_names, embeddings):
            self.name_to_embedding[name] = embedding.astype(np.float32)

        logger.info(f"✓ NameEmbedder fitted on {len(unique_names)} names")

    def encode(self, name: str) -> np.ndarray:
        """
        Encode a single name.

        Args:
            name: Node name

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Return cached if available
        if name in self.name_to_embedding:
            return self.name_to_embedding[name]

        # Not fitted yet or empty name
        if not name or self.vectorizer is None or self.svd is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Transform unseen name
        try:
            tfidf = self.vectorizer.transform([name])
            embedding = self.svd.transform(tfidf)[0]

            # Pad if needed
            if len(embedding) < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])

            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to encode name '{name}': {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def encode_batch(self, names: List[str]) -> np.ndarray:
        """
        Encode multiple names.

        Args:
            names: List of node names

        Returns:
            Embeddings array of shape (n_names, embedding_dim)
        """
        embeddings = [self.encode(name) for name in names]
        return np.stack(embeddings, axis=0)


def build_name_embedder(all_nodes: List, embedding_dim: int = 32) -> NameEmbedder:
    """
    Build a name embedder from all nodes.

    Args:
        all_nodes: List of all NodeInfo objects
        embedding_dim: Dimension for name embeddings

    Returns:
        Fitted NameEmbedder
    """
    from src.preprocessing.node_extraction import parse_name_prefix  # Import here to avoid circular dependency

    logger.info("Building name embedder from nodes...")

    # Extract all unique names (suffixes only, without prefix)
    names = set()
    for node in all_nodes:
        task_data = node.task_data
        name = task_data.get('name', '')
        if name:
            _, suffix = parse_name_prefix(name)
            if suffix:
                names.add(suffix)

    names_list = list(names)
    logger.info(f"Found {len(names_list)} unique node names")

    # Build and fit embedder
    embedder = NameEmbedder(embedding_dim=embedding_dim)
    embedder.fit(names_list)

    return embedder
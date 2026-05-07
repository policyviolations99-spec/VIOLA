"""
System prompt caching and role signature extraction.

This module handles:
1. Collecting unique system prompts across all traces
2. Extracting role signatures using TF-IDF
3. Caching embeddings for efficiency
"""

import logging
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class PromptCache:
    """
    Cache for system prompts and their role signature embeddings.
    """

    def __init__(self, cache_dir: Path, role_signature_dim: int = 32):
        """
        Initialize prompt cache.

        Args:
            cache_dir: Directory to store cache files
            role_signature_dim: Dimension for role signature vectors
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.role_signature_dim = role_signature_dim

        # Storage
        self.unique_prompts: Dict[str, str] = {}  # hash -> prompt text
        self.role_signatures: Dict[str, np.ndarray] = {}  # hash -> signature vector

        # TF-IDF components
        self.tfidf_vectorizer = None
        self.svd = None

        self.cache_file = self.cache_dir / "prompt_cache.pkl"

        logger.info(f"Initialized PromptCache (dim={role_signature_dim})")

    def _hash_prompt(self, prompt: str) -> str:
        """
        Create a hash for a prompt.

        Args:
            prompt: System prompt text

        Returns:
            Hash string
        """
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]

    def add_prompt(self, prompt: str) -> str:
        """
        Add a prompt to the cache.

        Args:
            prompt: System prompt text

        Returns:
            Hash of the prompt
        """
        prompt_hash = self._hash_prompt(prompt)

        if prompt_hash not in self.unique_prompts:
            self.unique_prompts[prompt_hash] = prompt

        return prompt_hash

    def collect_prompts_from_nodes(self, nodes: List) -> None:
        """
        Collect all unique system prompts from LLM nodes.

        Args:
            nodes: List of NodeInfo objects
        """
        logger.info("Collecting unique system prompts from nodes...")

        for node in nodes:
            if node.node_type == 'llm_call':
                task_data = node.task_data

                # Extract system prompt from input
                input_data = task_data.input
                system_prompt = input_data.get('gen_ai.prompt.0.content')

                if system_prompt:
                    self.add_prompt(system_prompt)

        logger.info(f"✓ Collected {len(self.unique_prompts)} unique system prompts")

    def _extract_role_keywords(self, prompts: List[str], top_n: int = 50) -> List[str]:
        """
        Extract important keywords from prompts using TF-IDF.

        Focus on verbs and nouns that indicate agent capabilities.

        Args:
            prompts: List of prompt texts
            top_n: Number of top keywords to extract

        Returns:
            List of important keywords
        """
        # Use TF-IDF to find distinctive terms
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{3,}\b',  # Words with 3+ letters
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_df=0.8,  # Ignore very common terms
            min_df=1
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(prompts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # Get average TF-IDF scores across all documents
        avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_indices = avg_scores.argsort()[-top_n:][::-1]

        top_keywords = [feature_names[i] for i in top_indices]

        logger.info(f"✓ Extracted {len(top_keywords)} key terms from prompts")
        logger.debug(f"Top keywords: {top_keywords[:10]}")

        return top_keywords

    def build_role_signatures(self) -> None:
        """
        Build role signature vectors for all cached prompts.

        Uses TF-IDF + SVD to create compact role representations.
        """
        logger.info("Building role signature vectors...")

        if not self.unique_prompts:
            logger.warning("No prompts to build signatures from")
            return

        # Get prompts in consistent order
        prompt_hashes = sorted(self.unique_prompts.keys())
        prompts = [self.unique_prompts[h] for h in prompt_hashes]

        # Build TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{3,}\b',
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=1
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(prompts)
        logger.info(f"  - TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Apply SVD to reduce to target dimension
        n_components = min(self.role_signature_dim, tfidf_matrix.shape[1], tfidf_matrix.shape[0])
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

        reduced_matrix = self.svd.fit_transform(tfidf_matrix)
        logger.info(f"  - SVD reduced to: {reduced_matrix.shape}")

        # Pad to exact dimension if needed
        if reduced_matrix.shape[1] < self.role_signature_dim:
            padding = np.zeros((reduced_matrix.shape[0], self.role_signature_dim - reduced_matrix.shape[1]))
            reduced_matrix = np.hstack([reduced_matrix, padding])

        # Store signatures
        for i, prompt_hash in enumerate(prompt_hashes):
            self.role_signatures[prompt_hash] = reduced_matrix[i].astype(np.float32)

        logger.info(f"✓ Built {len(self.role_signatures)} role signatures ({self.role_signature_dim}D)")

    def get_role_signature(self, prompt: str) -> np.ndarray:
        """
        Get role signature for a prompt.

        Args:
            prompt: System prompt text

        Returns:
            Role signature vector (role_signature_dim,)
        """
        prompt_hash = self._hash_prompt(prompt)

        if prompt_hash in self.role_signatures:
            return self.role_signatures[prompt_hash]
        else:
            # Prompt not in cache, return zero vector
            logger.warning(f"Prompt not in cache, returning zero vector")
            return np.zeros(self.role_signature_dim, dtype=np.float32)

    def save(self) -> None:
        """Save cache to disk."""
        cache_data = {
            'unique_prompts': self.unique_prompts,
            'role_signatures': self.role_signatures,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'svd': self.svd,
            'role_signature_dim': self.role_signature_dim
        }

        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"✓ Saved prompt cache to {self.cache_file}")

    def load(self) -> bool:
        """
        Load cache from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.cache_file.exists():
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self.unique_prompts = cache_data['unique_prompts']
            self.role_signatures = cache_data['role_signatures']
            self.tfidf_vectorizer = cache_data['tfidf_vectorizer']
            self.svd = cache_data['svd']
            self.role_signature_dim = cache_data['role_signature_dim']

            logger.info(f"✓ Loaded prompt cache from {self.cache_file}")
            logger.info(f"  - {len(self.unique_prompts)} unique prompts")
            logger.info(f"  - {len(self.role_signatures)} role signatures")

            return True

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    def __len__(self) -> int:
        """Return number of cached prompts."""
        return len(self.unique_prompts)


def build_prompt_cache(
        all_nodes: List,
        cache_dir: Path,
        role_signature_dim: int = 32
) -> PromptCache:
    """
    Build prompt cache from all nodes.

    This is used in Pass 1 of preprocessing.

    Args:
        all_nodes: List of all NodeInfo objects from all traces
        cache_dir: Directory to store cache
        role_signature_dim: Dimension for role signatures

    Returns:
        Built PromptCache
    """
    logger.info("=" * 80)
    logger.info("PASS 1: Building Prompt Cache")
    logger.info("=" * 80)

    cache = PromptCache(cache_dir, role_signature_dim)

    # Collect prompts
    cache.collect_prompts_from_nodes(all_nodes)

    # Build signatures
    cache.build_role_signatures()

    # Save cache
    cache.save()

    logger.info("=" * 80)
    logger.info("PASS 1 Complete")
    logger.info("=" * 80)

    return cache
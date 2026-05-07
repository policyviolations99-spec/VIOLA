"""
Configuration for the cross-model validation experiment.

Paths are configurable via environment variables so the public release does
not bake in any author-local filesystem layout.

Required env:
    BENCHMARK_RUN_DIR     — directory containing manifest.json and logs/
    GPT_VALIDATION_DIR    — directory containing per-trace _validation.json files
                            produced by the existing GPT-4.1 pipeline.

Optional env:
    CROSS_MODEL_OUTPUT_DIR  — where to write results/ and paper_tables/
                               (default: alongside this package).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
# All three judges are reachable through a single OpenAI-compatible endpoint
# (e.g. an IBM litellm proxy or OpenRouter) — only the model identifier
# changes between runners. Override these via env if your gateway exposes the
# models under different names.

GPT_MODEL_ID = os.environ.get("CROSS_MODEL_GPT_ID", "Azure/gpt-4.1")
CLAUDE_MODEL = os.environ.get("CROSS_MODEL_CLAUDE_ID", "claude-sonnet-4-6")
GEMINI_MODEL = os.environ.get("CROSS_MODEL_GEMINI_ID", "GCP/gemini-2.5-pro")

JUDGE_MODELS = [GPT_MODEL_ID, CLAUDE_MODEL, GEMINI_MODEL]

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
# Credentials use the same JUDGE_BASE_URL / JUDGE_API_KEY (or
# OPENAI_BASE_URL / OPENAI_API_KEY) variables the existing GPT pipeline
# consumes. No provider-specific keys needed.

# ---------------------------------------------------------------------------
# Inference parameters (mirror the existing GPT-4.1 judge configuration)
# ---------------------------------------------------------------------------
TEMPERATURE = 0.0
MAX_TOKENS = 1024
N_RUNS_PER_JUDGE = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)
MAX_CONCURRENT_PER_PROVIDER = 10

# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------
TARGET_SAMPLE_SIZE = 150
N_ACCEPTED_PER_VIOLATION_TYPE = 10
N_REJECTED_TRACES = 30
N_CLEAN_CONTROLS = 10
SAMPLING_SEED = 42

ACTIVE_VIOLATION_TYPES = (
    "V1", "V2", "V3a", "V3b", "V4", "V6",
    "V7", "V8", "V9", "V10", "V11",
)

CONSENSUS_THRESHOLD = 2 / 3

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_DIR = PACKAGE_DIR


def _output_dir() -> Path:
    return Path(os.environ.get("CROSS_MODEL_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR))


def results_dir() -> Path:
    return _output_dir() / "results"


def paper_tables_dir() -> Path:
    return _output_dir() / "paper_tables"


def judgments_path() -> Path:
    return results_dir() / "cross_model_judgments.jsonl"


def aggregated_labels_path() -> Path:
    return results_dir() / "aggregated_labels.csv"


def sample_manifest_path() -> Path:
    return results_dir() / "sample_manifest.json"


def benchmark_run_dir() -> Path:
    p = os.environ.get("BENCHMARK_RUN_DIR")
    if not p:
        raise RuntimeError(
            "BENCHMARK_RUN_DIR is not set. Point it at the directory containing "
            "manifest.json and logs/ for the dataset you want to validate."
        )
    return Path(p)


def gpt_validation_dir() -> Path:
    p = os.environ.get("GPT_VALIDATION_DIR")
    if not p:
        raise RuntimeError(
            "GPT_VALIDATION_DIR is not set. Point it at the directory containing "
            "the per-trace _validation.json files produced by the existing "
            "GPT-4.1 judge pipeline."
        )
    return Path(p)

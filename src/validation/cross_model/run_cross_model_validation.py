"""
Cross-model validation entry point (public release).

Re-runs the existing J1 (theoretic) and J2 (executional) judges on a stratified
subset of the benchmark using two additional model families — Claude Sonnet
4.6 and Gemini 2.5 Pro — and reports inter-judge agreement metrics for the
paper appendix.

Required environment:
    BENCHMARK_RUN_DIR     — directory with manifest.json + logs/ (the dataset).
    GPT_VALIDATION_DIR    — directory with per-trace _validation.json files
                            produced by the existing GPT-4.1 judge pipeline.
    ANTHROPIC_API_KEY     — Anthropic API key for Claude.
    GOOGLE_API_KEY        — Google AI Studio key for Gemini (or GEMINI_API_KEY).

Usage:
    python -m src.validation.cross_model.run_cross_model_validation
    python -m src.validation.cross_model.run_cross_model_validation --sample-only
    python -m src.validation.cross_model.run_cross_model_validation --metrics-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Make ``src.validation`` importable when invoked as ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.validation.models import TraceForValidation  # noqa: E402

from .config import (  # noqa: E402
    JUDGE_MODELS,
    aggregated_labels_path,
    judgments_path,
    paper_tables_dir,
    sample_manifest_path,
)
from .metrics import (  # noqa: E402
    acceptance_overlap,
    aggregate_to_consensus,
    merge_gpt_consensus,
    pairwise_kappa,
    per_violation_type_kappa,
    reclassification_agreement,
    three_way_fleiss,
)
from .orchestrator import load_all_judgments, orchestrate  # noqa: E402
from .reporting import generate_paper_tables  # noqa: E402
from .sampling import build_stratified_sample, write_manifest  # noqa: E402
from .trace_loader import (  # noqa: E402
    GPTLabel,
    load_clean_control_traces,
    load_gpt_labels,
    load_rejected_traces,
    load_violation_traces,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cross_model")


def _load_sample(args) -> tuple[List[TraceForValidation], Dict[str, GPTLabel]]:
    logger.info("Loading GPT-4.1 labels …")
    gpt_labels = load_gpt_labels()
    logger.info("Loading violation traces …")
    accepted = load_violation_traces()
    accepted = [t for t in accepted if gpt_labels.get(t.run_id) and gpt_labels[t.run_id].accepted]
    logger.info("Loading rejected traces …")
    rejected = load_rejected_traces(gpt_labels)
    logger.info("Loading clean control traces …")
    clean = load_clean_control_traces()

    sample = build_stratified_sample(accepted, rejected, clean, gpt_labels)
    if args.max_traces:
        sample = sample[: args.max_traces]
        logger.info("Truncated sample to %d traces", len(sample))

    write_manifest(sample, gpt_labels)
    return sample, gpt_labels


async def _orchestrate_async(
    sample: List[TraceForValidation],
    gpt_labels: Dict[str, GPTLabel],
    args,
) -> None:
    runners = []
    if not args.skip_claude:
        from .judge_runners import ClaudeJudgeRunner
        runners.append(ClaudeJudgeRunner())
    if not args.skip_gemini:
        from .judge_runners import GeminiJudgeRunner
        runners.append(GeminiJudgeRunner())

    # Back-fill GPT-4.1 on traces missing labels (e.g. clean controls) so the
    # agreement tables cover the full sample three-way.
    missing_gpt = [t for t in sample if t.run_id not in gpt_labels]
    if missing_gpt and not args.skip_gpt_backfill:
        from .judge_runners import GPTJudgeRunner
        logger.info("Back-filling GPT-4.1 on %d traces missing labels.",
                    len(missing_gpt))
        await orchestrate(missing_gpt, [GPTJudgeRunner()])

    if not runners:
        logger.warning("No alt runners selected; skipping cross-model orchestration.")
        return
    await orchestrate(sample, runners)


def _compute_metrics(
    sample: List[TraceForValidation], gpt_labels: Dict[str, GPTLabel],
) -> None:
    import pandas as pd

    judgments = load_all_judgments()
    if not judgments:
        logger.error("No judgments found at %s — run orchestration first.",
                     judgments_path())
        return

    consensus_new = aggregate_to_consensus(judgments)
    sampled_ids = {t.run_id for t in sample}
    gpt_for_sample = {rid: lbl for rid, lbl in gpt_labels.items() if rid in sampled_ids}
    consensus = merge_gpt_consensus(consensus_new, gpt_for_sample)

    out_path = aggregated_labels_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    consensus.to_csv(out_path, index=False)
    logger.info("Wrote aggregated labels: %s", out_path)

    sample_meta = pd.DataFrame([
        {
            "trace_id": t.run_id,
            "violation_id": t.violation_id,
            "target_agent": t.target_agent,
            "gpt_accepted": gpt_labels[t.run_id].accepted if t.run_id in gpt_labels else None,
            "gpt_final_label": gpt_labels[t.run_id].final_label if t.run_id in gpt_labels else None,
        }
        for t in sample
    ])

    pw_j1 = pairwise_kappa(consensus, JUDGE_MODELS, "j1")
    pw_j2 = pairwise_kappa(consensus, JUDGE_MODELS, "j2")
    fleiss_j1 = three_way_fleiss(consensus, JUDGE_MODELS, "j1")
    fleiss_j2 = three_way_fleiss(consensus, JUDGE_MODELS, "j2")
    per_type = per_violation_type_kappa(consensus, sample_meta, JUDGE_MODELS, "j2")
    overlap = acceptance_overlap(consensus, JUDGE_MODELS)
    rec = reclassification_agreement(consensus, JUDGE_MODELS)

    print("\n=== Pairwise Kappa (J1) ===")
    print(pw_j1.to_string(index=False))
    print("\n=== Pairwise Kappa (J2) ===")
    print(pw_j2.to_string(index=False))
    print(f"\n=== Fleiss J1 === {fleiss_j1}")
    print(f"=== Fleiss J2 === {fleiss_j2}")
    print("\n=== Acceptance Overlap ===")
    print(overlap.to_string(index=False))
    print("\n=== Reclassification Agreement ===")
    print(rec.to_string(index=False))

    generate_paper_tables(
        pw_j1, pw_j2, fleiss_j1, fleiss_j2,
        per_type, overlap, rec,
        judgments, consensus, sample_meta,
    )
    logger.info("Paper tables written under %s", paper_tables_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-only", action="store_true",
                        help="Build the sample manifest and exit.")
    parser.add_argument("--skip-claude", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-gpt-backfill", action="store_true",
                        help="Don't run GPT-4.1 on traces missing labels (clean controls).")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Skip orchestration; just (re-)compute metrics.")
    parser.add_argument("--max-traces", type=int, default=None,
                        help="Cap sample size (for smoke tests).")
    args = parser.parse_args()

    sample, gpt_labels = _load_sample(args)
    logger.info("Sample size: %d", len(sample))

    if args.sample_only:
        return

    if not args.metrics_only:
        asyncio.run(_orchestrate_async(sample, gpt_labels, args))

    _compute_metrics(sample, gpt_labels)
    logger.info("Done. Manifest: %s   judgments: %s   tables: %s",
                sample_manifest_path(), judgments_path(), paper_tables_dir())


if __name__ == "__main__":
    main()

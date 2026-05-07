"""
Paper-ready outputs: LaTeX tables and a markdown disagreement appendix.

The functions here consume the DataFrames produced by ``metrics.py`` and write
files into ``cross_model/paper_tables/``. Every table is wrapped in
``\\begin{tabular}{...}`` so it can be dropped straight into a paper appendix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import JUDGE_MODELS, paper_tables_dir

logger = logging.getLogger(__name__)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    logger.info("Wrote %s", path)


def _fmt_kappa_with_ci(row: pd.Series) -> str:
    if pd.isna(row["kappa"]):
        return "n/a"
    if pd.isna(row.get("ci_low", float("nan"))):
        return f"{row['kappa']:.2f}"
    return f"{row['kappa']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"


# ---------------------------------------------------------------------------
# Pairwise kappa (J1 + J2 in one table)
# ---------------------------------------------------------------------------

def write_pairwise_kappa_table(
    pw_j1: pd.DataFrame, pw_j2: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    """One row per judge pair, columns J1 / J2 (kappa with 95% CI)."""
    out_path = out_path or paper_tables_dir() / "pairwise_kappa.tex"
    pairs = sorted({(r["judge_a"], r["judge_b"]) for _, r in pw_j1.iterrows()} |
                   {(r["judge_a"], r["judge_b"]) for _, r in pw_j2.iterrows()})

    def lookup(df: pd.DataFrame, a: str, b: str) -> str:
        rows = df[(df["judge_a"] == a) & (df["judge_b"] == b)]
        if rows.empty:
            return "—"
        return _fmt_kappa_with_ci(rows.iloc[0])

    lines = [
        "% Pairwise Cohen's kappa with 95% bootstrap CI for J1 (theoretic) and J2 (executional)",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Judge pair & J1 ($\kappa$ [95\% CI]) & J2 ($\kappa$ [95\% CI]) \\",
        r"\midrule",
    ]
    for a, b in pairs:
        lines.append(f"{a} vs.\\ {b} & {lookup(pw_j1, a, b)} & {lookup(pw_j2, a, b)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Fleiss' kappa (three-way)
# ---------------------------------------------------------------------------

def write_fleiss_table(
    fleiss_j1: dict, fleiss_j2: dict,
    out_path: Path | None = None,
) -> None:
    out_path = out_path or paper_tables_dir() / "fleiss_kappa.tex"

    def fmt(d: dict) -> str:
        if "error" in d:
            return f"n/a ({d['error']})"
        if pd.isna(d.get("kappa", float("nan"))):
            return "n/a"
        return f"{d['kappa']:.2f} (n={d['n']})"

    lines = [
        "% Three-way Fleiss' kappa across GPT-4.1, Claude Sonnet 4.6, Gemini 2.5 Pro",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Stage & Fleiss $\kappa$ (n) \\",
        r"\midrule",
        f"J1 (theoretic) & {fmt(fleiss_j1)} \\\\",
        f"J2 (executional) & {fmt(fleiss_j2)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _write(out_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Per-violation-type kappa (J2)
# ---------------------------------------------------------------------------

def write_per_type_kappa_table(
    per_type: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    out_path = out_path or paper_tables_dir() / "per_type_kappa.tex"
    if per_type.empty:
        _write(out_path, "% No data\n")
        return
    pivot = per_type.pivot_table(
        index="violation_id",
        columns=["judge_a", "judge_b"],
        values="kappa",
        aggfunc="first",
    )
    counts = per_type.pivot_table(
        index="violation_id",
        columns=["judge_a", "judge_b"],
        values="n",
        aggfunc="first",
    )

    pair_cols = list(pivot.columns)
    header = "Violation & " + " & ".join(f"{a} vs.\\ {b}" for a, b in pair_cols) + r" \\"
    lines = [
        "% Pairwise Cohen's kappa per violation type (J2 stage)",
        r"\begin{tabular}{l" + "c" * len(pair_cols) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for vtype in pivot.index:
        cells = []
        for col in pair_cols:
            k = pivot.loc[vtype, col]
            n = counts.loc[vtype, col] if col in counts.columns else None
            if pd.isna(k):
                cells.append("—")
            else:
                cells.append(f"{k:.2f} (n={int(n) if pd.notna(n) else '?'})")
        lines.append(f"{vtype} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Acceptance overlap
# ---------------------------------------------------------------------------

def write_acceptance_overlap_table(
    overlap: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    out_path = out_path or paper_tables_dir() / "acceptance_overlap.tex"
    if overlap.empty:
        _write(out_path, "% No data\n")
        return
    lines = [
        "% Acceptance-decision overlap per judge pair",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Judge A & Judge B & Both accept & Both reject & A only & B only & Agreement & n \\",
        r"\midrule",
    ]
    for _, r in overlap.iterrows():
        lines.append(
            f"{r['judge_a']} & {r['judge_b']} & "
            f"{int(r['both_accept'])} & {int(r['both_reject'])} & "
            f"{int(r['a_only_accept'])} & {int(r['b_only_accept'])} & "
            f"{r['agreement_rate'] * 100:.1f}\\% & {int(r['n'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_path, "\n".join(lines) + "\n")


def write_reclassification_table(
    rec: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    out_path = out_path or paper_tables_dir() / "reclassification_agreement.tex"
    if rec.empty:
        _write(out_path, "% No data\n")
        return
    lines = [
        "% Hard/soft label agreement on doubly-accepted traces",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Judge A & Judge B & Both accepted (n) & Label agree & Agreement \\",
        r"\midrule",
    ]
    for _, r in rec.iterrows():
        lines.append(
            f"{r['judge_a']} & {r['judge_b']} & "
            f"{int(r['n'])} & {int(r['agree_count'])} & "
            f"{r['agree_rate'] * 100:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Disagreement examples (markdown)
# ---------------------------------------------------------------------------

def find_disagreements(
    consensus: pd.DataFrame,
    judges: List[str] = list(JUDGE_MODELS),
    top_n: int = 10,
) -> pd.DataFrame:
    """Pick traces where judges disagree most on J2 consensus."""
    pivot = consensus.pivot_table(
        index="trace_id", columns="judge_model",
        values="j2_consensus", aggfunc="first",
    )
    judges = [j for j in judges if j in pivot.columns]
    pivot = pivot[judges].dropna()

    def diversity(row):
        return len({str(v) for v in row})

    pivot["diversity"] = pivot.apply(diversity, axis=1)
    pivot = pivot.sort_values("diversity", ascending=False).head(top_n)
    return pivot


def write_disagreement_examples(
    disagreements: pd.DataFrame,
    judgments: List[dict],
    sample_meta: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    out_path = out_path or paper_tables_dir() / "disagreement_examples.md"
    if disagreements.empty:
        _write(out_path, "# No disagreements found\n")
        return

    by_trace_judge_stage: Dict[tuple, List[dict]] = {}
    for r in judgments:
        key = (r.get("trace_id"), r.get("judge_model"), r.get("stage"))
        by_trace_judge_stage.setdefault(key, []).append(r)

    meta_lookup = sample_meta.set_index("trace_id").to_dict(orient="index")

    lines = ["# Cross-model disagreement examples\n"]
    for trace_id, row in disagreements.iterrows():
        meta = meta_lookup.get(trace_id, {})
        lines.append(f"## {trace_id}")
        lines.append(f"- violation_id: `{meta.get('violation_id', '?')}`")
        lines.append(f"- target_agent: `{meta.get('target_agent', '?')}`")
        lines.append(f"- gpt_accepted: `{meta.get('gpt_accepted')}`")
        lines.append(f"- gpt_final_label: `{meta.get('gpt_final_label')}`\n")

        for judge in row.index:
            if judge == "diversity":
                continue
            lines.append(f"### Judge: {judge}")
            for stage in ("j1", "j2"):
                runs = by_trace_judge_stage.get((trace_id, judge, stage), [])
                verdicts = [r.get("judgment") for r in runs]
                lines.append(f"- **{stage.upper()}** verdicts: `{verdicts}`")
                if runs:
                    sample_run = runs[0]
                    reason = sample_run.get("reasoning") or sample_run.get("evidence") or ""
                    if reason:
                        snippet = str(reason).strip().replace("\n", " ")
                        if len(snippet) > 400:
                            snippet = snippet[:400] + "…"
                        lines.append(f"  - _example reasoning_: {snippet}")
            lines.append("")
        lines.append("---\n")

    _write(out_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def generate_paper_tables(
    pw_j1: pd.DataFrame, pw_j2: pd.DataFrame,
    fleiss_j1: dict, fleiss_j2: dict,
    per_type: pd.DataFrame,
    overlap: pd.DataFrame, rec: pd.DataFrame,
    judgments: List[dict], consensus: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> None:
    """Write every paper-ready artifact."""
    write_pairwise_kappa_table(pw_j1, pw_j2)
    write_fleiss_table(fleiss_j1, fleiss_j2)
    write_per_type_kappa_table(per_type)
    write_acceptance_overlap_table(overlap)
    write_reclassification_table(rec)
    disagreements = find_disagreements(consensus)
    write_disagreement_examples(disagreements, judgments, sample_meta)

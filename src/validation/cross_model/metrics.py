"""
Inter-judge agreement metrics.

We support pairwise Cohen's kappa with bootstrap CIs, three-way Fleiss' kappa,
per-violation-type kappa, and acceptance-decision overlap. The aggregation
helper folds raw per-run records into one consensus row per (trace, judge).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .config import CONSENSUS_THRESHOLD, JUDGE_MODELS

try:
    from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss_kappa
except ImportError:  # pragma: no cover — handled at runtime
    _fleiss_kappa = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consensus aggregation per (trace, judge)
# ---------------------------------------------------------------------------

VIOLATION_LABELS = ("hard_violation", "soft_violation")


def aggregate_to_consensus(
    judgments: List[dict],
    threshold: float = CONSENSUS_THRESHOLD,
) -> pd.DataFrame:
    """Reduce per-run judgments to one consensus row per (trace, judge_model).

    Returns columns:
      trace_id, judge_model, violation_id, target_agent,
      j1_consensus, j2_consensus, accepted, hard_soft.
    """
    df = pd.DataFrame(judgments)
    if df.empty:
        return pd.DataFrame(columns=[
            "trace_id", "judge_model",
            "j1_consensus", "j2_consensus", "accepted", "hard_soft",
        ])

    records = []
    grouped = df.groupby(["trace_id", "judge_model"], dropna=False)
    for (trace_id, judge), grp in grouped:
        j1 = grp[grp["stage"] == "j1"]["judgment"].tolist()
        j2 = grp[grp["stage"] == "j2"]["judgment"].tolist()

        # J1: majority is policy_absent?
        n_j1 = len(j1) or 1
        j1_absent = sum(1 for x in j1 if x == "policy_absent")
        j1_consensus = "policy_absent" if (j1_absent / n_j1) >= threshold else "policy_present"

        # J2: majority is any kind of violation?
        n_j2 = len(j2) or 1
        violation_runs = [x for x in j2 if x in VIOLATION_LABELS]
        if (len(violation_runs) / n_j2) >= threshold and violation_runs:
            label = Counter(violation_runs).most_common(1)[0][0]
            j2_consensus = label
            hard_soft = "hard" if label == "hard_violation" else "soft"
        else:
            j2_consensus = "no_violation"
            hard_soft = None

        accepted = (j1_consensus == "policy_absent") and (j2_consensus != "no_violation")
        records.append({
            "trace_id": trace_id,
            "judge_model": judge,
            "j1_consensus": j1_consensus,
            "j2_consensus": j2_consensus,
            "accepted": accepted,
            "hard_soft": hard_soft,
        })
    return pd.DataFrame(records)


def merge_gpt_consensus(consensus: pd.DataFrame, gpt_labels: dict) -> pd.DataFrame:
    """Append synthetic GPT-4.1 consensus rows from existing labels.

    ``gpt_labels`` is dict[run_id -> trace_loader.GPTLabel] for the sampled
    traces. Traces whose GPT consensus is already in ``consensus`` (because
    we re-ran GPT-4.1 in the orchestration step) are skipped to avoid
    duplicate rows.
    """
    from .config import GPT_MODEL_ID

    if not consensus.empty:
        already = set(consensus.loc[consensus["judge_model"] == GPT_MODEL_ID,
                                    "trace_id"].tolist())
    else:
        already = set()

    rows = []
    for run_id, lbl in gpt_labels.items():
        if run_id in already:
            continue
        rows.append({
            "trace_id": run_id,
            "judge_model": GPT_MODEL_ID,
            "j1_consensus": lbl.judge1_consensus or "n/a",
            "j2_consensus": lbl.judge2_consensus or "n/a",
            "accepted": bool(lbl.accepted),
            "hard_soft": lbl.final_label,
        })
    if not rows:
        return consensus
    return pd.concat([consensus, pd.DataFrame(rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# Kappa metrics
# ---------------------------------------------------------------------------

def _bootstrap_kappa_ci(
    a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_samples = len(a)
    if n_samples < 2:
        return float("nan"), float("nan")
    boots = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        try:
            boots.append(cohen_kappa_score(a[idx], b[idx]))
        except ValueError:
            continue
    if not boots:
        return float("nan"), float("nan")
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def pairwise_kappa(
    consensus: pd.DataFrame,
    judges: Iterable[str] = JUDGE_MODELS,
    stage: str = "j1",
) -> pd.DataFrame:
    """Compute pairwise Cohen's kappa for a single stage."""
    column = f"{stage}_consensus"
    pivot = consensus.pivot_table(
        index="trace_id", columns="judge_model", values=column, aggfunc="first",
    )
    judges = [j for j in judges if j in pivot.columns]

    rows = []
    for i, j_a in enumerate(judges):
        for j_b in judges[i + 1:]:
            mask = pivot[j_a].notna() & pivot[j_b].notna()
            a = pivot.loc[mask, j_a].astype(str).values
            b = pivot.loc[mask, j_b].astype(str).values
            n = len(a)
            if n < 2 or len(set(a) | set(b)) < 2:
                kappa, ci_low, ci_high = float("nan"), float("nan"), float("nan")
            else:
                kappa = float(cohen_kappa_score(a, b))
                ci_low, ci_high = _bootstrap_kappa_ci(a, b)
            rows.append({
                "judge_a": j_a, "judge_b": j_b, "stage": stage,
                "kappa": kappa, "ci_low": ci_low, "ci_high": ci_high, "n": n,
            })
    return pd.DataFrame(rows)


def three_way_fleiss(
    consensus: pd.DataFrame,
    judges: Iterable[str] = JUDGE_MODELS,
    stage: str = "j1",
) -> dict:
    """Fleiss' kappa across all three judges for a single stage."""
    if _fleiss_kappa is None:
        return {"stage": stage, "kappa": float("nan"), "n": 0,
                "error": "statsmodels not installed"}
    column = f"{stage}_consensus"
    pivot = consensus.pivot_table(
        index="trace_id", columns="judge_model", values=column, aggfunc="first",
    )
    judges = [j for j in judges if j in pivot.columns]
    if len(judges) < 3:
        return {"stage": stage, "kappa": float("nan"), "n": 0,
                "judges": judges, "error": "fewer than 3 judges available"}
    pivot = pivot[judges].dropna()
    if pivot.empty:
        return {"stage": stage, "kappa": float("nan"), "n": 0}
    categories = sorted({v for col in pivot.columns for v in pivot[col].unique()})
    counts = np.zeros((len(pivot), len(categories)), dtype=int)
    for i, row in enumerate(pivot.values):
        for j, cat in enumerate(categories):
            counts[i, j] = int((row == cat).sum())
    return {"stage": stage,
            "kappa": float(_fleiss_kappa(counts)),
            "n": int(len(pivot)),
            "judges": judges,
            "categories": categories}


def per_violation_type_kappa(
    consensus: pd.DataFrame,
    sample_meta: pd.DataFrame,
    judges: Iterable[str] = JUDGE_MODELS,
    stage: str = "j2",
    min_n: int = 5,
) -> pd.DataFrame:
    """Pairwise kappa broken out by violation_id."""
    column = f"{stage}_consensus"
    df = consensus.merge(sample_meta[["trace_id", "violation_id"]], on="trace_id")
    rows = []
    judges = list(judges)
    for vtype, grp in df.groupby("violation_id"):
        pivot = grp.pivot_table(
            index="trace_id", columns="judge_model", values=column, aggfunc="first",
        )
        present = [j for j in judges if j in pivot.columns]
        for i, j_a in enumerate(present):
            for j_b in present[i + 1:]:
                mask = pivot[j_a].notna() & pivot[j_b].notna()
                if mask.sum() < min_n:
                    continue
                a = pivot.loc[mask, j_a].astype(str).values
                b = pivot.loc[mask, j_b].astype(str).values
                if len(set(a) | set(b)) < 2:
                    continue
                rows.append({
                    "violation_id": vtype, "judge_a": j_a, "judge_b": j_b,
                    "stage": stage, "kappa": float(cohen_kappa_score(a, b)),
                    "n": int(mask.sum()),
                })
    return pd.DataFrame(rows)


def acceptance_overlap(
    consensus: pd.DataFrame, judges: Iterable[str] = JUDGE_MODELS,
) -> pd.DataFrame:
    """Confusion of acceptance decisions for each judge pair."""
    pivot = consensus.pivot_table(
        index="trace_id", columns="judge_model", values="accepted", aggfunc="first",
    )
    judges = [j for j in judges if j in pivot.columns]
    rows = []
    for i, j_a in enumerate(judges):
        for j_b in judges[i + 1:]:
            mask = pivot[j_a].notna() & pivot[j_b].notna()
            sub = pivot.loc[mask, [j_a, j_b]].astype(bool)
            both_acc = int(((sub[j_a]) & (sub[j_b])).sum())
            both_rej = int(((~sub[j_a]) & (~sub[j_b])).sum())
            a_only = int(((sub[j_a]) & (~sub[j_b])).sum())
            b_only = int(((~sub[j_a]) & (sub[j_b])).sum())
            total = both_acc + both_rej + a_only + b_only
            agree = (both_acc + both_rej) / total if total else float("nan")
            rows.append({
                "judge_a": j_a, "judge_b": j_b,
                "both_accept": both_acc, "both_reject": both_rej,
                "a_only_accept": a_only, "b_only_accept": b_only,
                "agreement_rate": agree, "n": total,
            })
    return pd.DataFrame(rows)


def reclassification_agreement(
    consensus: pd.DataFrame, judges: Iterable[str] = JUDGE_MODELS,
) -> pd.DataFrame:
    """For doubly-accepted traces, fraction where hard/soft labels match."""
    pivot_acc = consensus.pivot_table(
        index="trace_id", columns="judge_model", values="accepted", aggfunc="first",
    )
    pivot_lab = consensus.pivot_table(
        index="trace_id", columns="judge_model", values="hard_soft", aggfunc="first",
    )
    judges = [j for j in judges if j in pivot_acc.columns]
    rows = []
    for i, j_a in enumerate(judges):
        for j_b in judges[i + 1:]:
            # Cast to bool first (object dtype + fillna triggers a pandas FutureWarning).
            acc_a = pivot_acc[j_a].astype("boolean").fillna(False).astype(bool)
            acc_b = pivot_acc[j_b].astype("boolean").fillna(False).astype(bool)
            both = acc_a & acc_b
            # If a judge never produced a hard/soft label (all rows are None)
            # the column is dropped from pivot_lab — supply an empty Series.
            sub_a = pivot_lab[j_a] if j_a in pivot_lab.columns else pd.Series(index=pivot_acc.index, dtype=object)
            sub_b = pivot_lab[j_b] if j_b in pivot_lab.columns else pd.Series(index=pivot_acc.index, dtype=object)
            sub_a = sub_a.loc[both]
            sub_b = sub_b.loc[both]
            both_set = sub_a.notna() & sub_b.notna()
            n = int(both_set.sum())
            agree = int((sub_a[both_set] == sub_b[both_set]).sum())
            rows.append({
                "judge_a": j_a, "judge_b": j_b,
                "n": n,
                "agree_count": agree,
                "agree_rate": (agree / n) if n else float("nan"),
            })
    return pd.DataFrame(rows)

"""
Async orchestration of cross-model judge calls with on-disk caching.

Caching policy: every successfully completed (trace_id, judge_model, stage,
run_idx) record is appended to ``cross_model_judgments.jsonl``. On subsequent
runs, identical tuples are skipped, so a crashed or partial run can be resumed
by re-invoking the script.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from src.validation.models import TraceForValidation

from .config import (
    MAX_CONCURRENT_PER_PROVIDER,
    N_RUNS_PER_JUDGE,
    judgments_path,
)
from .judge_runners import JudgeRunner

logger = logging.getLogger(__name__)


CacheKey = Tuple[str, str, str, int]


def _cache_key(trace_id: str, model_id: str, stage: str, run_idx: int) -> CacheKey:
    return (trace_id, model_id, stage, run_idx)


def load_existing_cache(path: Path = None) -> Set[CacheKey]:
    path = path or judgments_path()
    keys: Set[CacheKey] = set()
    if not path.exists():
        return keys
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                keys.add(_cache_key(rec["trace_id"], rec["judge_model"],
                                    rec["stage"], rec["run_idx"]))
            except KeyError:
                continue
    return keys


def load_all_judgments(path: Path = None) -> List[dict]:
    path = path or judgments_path()
    if not path.exists():
        return []
    out: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


async def _run_one(
    runner: JudgeRunner,
    trace: TraceForValidation,
    stage: str,
    run_idx: int,
    semaphore: asyncio.Semaphore,
    cache_keys: Set[CacheKey],
    writer_lock: asyncio.Lock,
    out_path: Path,
) -> None:
    key = _cache_key(trace.run_id, runner.model_id, stage, run_idx)
    if key in cache_keys:
        return
    async with semaphore:
        if stage == "j1":
            record = await runner.run_j1(trace, run_idx)
        else:
            record = await runner.run_j2(trace, run_idx)
        record.setdefault("judge_model", runner.model_id)
    async with writer_lock:
        with out_path.open("a") as w:
            w.write(json.dumps(record, ensure_ascii=False) + "\n")
        cache_keys.add(key)


async def orchestrate(
    sample: Iterable[TraceForValidation],
    runners: Iterable[JudgeRunner],
    out_path: Path = None,
    n_runs: int = N_RUNS_PER_JUDGE,
) -> None:
    out_path = out_path or judgments_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_keys = load_existing_cache(out_path)
    logger.info("Cache contains %d completed calls.", len(cache_keys))

    runners = list(runners)
    semaphores = {r.model_id: asyncio.Semaphore(MAX_CONCURRENT_PER_PROVIDER) for r in runners}
    writer_lock = asyncio.Lock()

    sample_list = list(sample)
    total_planned = len(sample_list) * len(runners) * 2 * n_runs
    pending = total_planned - len(cache_keys)
    logger.info("Planned: %d  cached: %d  pending: %d",
                total_planned, len(cache_keys), pending)

    tasks = []
    for trace in sample_list:
        for runner in runners:
            for stage in ("j1", "j2"):
                for run_idx in range(n_runs):
                    tasks.append(asyncio.create_task(_run_one(
                        runner, trace, stage, run_idx,
                        semaphores[runner.model_id],
                        cache_keys, writer_lock, out_path,
                    )))

    if not tasks:
        return
    done = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        done += 1
        if done % 50 == 0 or done == len(tasks):
            logger.info("Completed %d/%d", done, len(tasks))

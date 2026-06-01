"""Scoring orchestration: validated submission + concept + day -> ScoreResult.

``score = hit_rate`` (§8 of the spec): the fraction of the day's completions that the
active concept's pinned detector flags positive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .detectors import get_detector
from .prompts import PromptItem, PromptPool
from .submission import Submission


@dataclass(frozen=True)
class CompletionRecord:
    id: int
    prompt: str
    completion: str
    hit: bool
    matched: list[str]


@dataclass(frozen=True)
class ScoreResult:
    score: float
    hit_count: int
    total: int
    per_completion: list[CompletionRecord] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def score_completions(
    completions: list[str],
    prompts: list[PromptItem],
    concept: str,
    settings: Settings,
    return_completions: bool = True,
) -> tuple[float, int, list[CompletionRecord]]:
    """Pure (no-GPU) scoring of pre-generated completions — reused by tests."""
    detector = get_detector(concept, settings.detectors)
    results = detector.detect_batch(completions)
    hit_count = sum(1 for r in results if r.hit)
    total = len(results)
    records: list[CompletionRecord] = []
    if return_completions:
        for item, comp, res in zip(prompts, completions, results):
            records.append(
                CompletionRecord(
                    id=item.id,
                    prompt=item.instruction,
                    completion=comp,
                    hit=res.hit,
                    matched=res.matched,
                )
            )
    score = hit_count / total if total else 0.0
    return score, hit_count, records


def score_submission(
    runtime,
    settings: Settings,
    submission: Submission,
    active_concept: str,
    day_index: int,
    seed: int,
    pool: PromptPool,
    return_completions: bool = True,
) -> ScoreResult:
    t0 = time.perf_counter()
    prompts = pool.sample_day(day_index, seed, settings.prompts.per_day)
    t_sample = time.perf_counter()

    completions = runtime.generate([p.instruction for p in prompts], submission)
    t_gen = time.perf_counter()

    score, hit_count, records = score_completions(
        completions, prompts, active_concept, settings, return_completions
    )
    t_detect = time.perf_counter()

    return ScoreResult(
        score=score,
        hit_count=hit_count,
        total=len(prompts),
        per_completion=records,
        diagnostics={
            "alpha": submission.alpha,
            "concept": active_concept,
            "layer": submission.layer,
            "day_index": day_index,
            "seed": seed,
            "detector_version": settings.detectors.get(active_concept),
            "model_revision": settings.model.revision,
            "timings_ms": {
                "sample": round((t_sample - t0) * 1000, 2),
                "generate": round((t_gen - t_sample) * 1000, 2),
                "detect": round((t_detect - t_gen) * 1000, 2),
            },
        },
    )

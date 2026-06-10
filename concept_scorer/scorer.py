"""Scoring orchestration: validated submission + concept + day -> ScoreResult.

Each concept's day-score is set by its ``scoring`` config: ``hit_rate`` (the fraction of the
day's completions the detector flags positive — spec §8) or ``graded`` (mean normalized
per-completion intensity, in [0,1]). ``graded`` is a deliberate, config-gated deviation from §8.
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
    # Raw per-completion detector intensity (summed cue weights / AFINN net valence).
    score: float
    matched: list[str]


@dataclass(frozen=True)
class ScoreResult:
    score: float
    hit_count: int
    total: int
    per_completion: list[CompletionRecord] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _aggregate(results, mode: str, saturation: float) -> float:
    """Day-score in [0,1]: hit-fraction (hit_rate) or mean clamped intensity (graded)."""
    if not results:
        return 0.0
    if mode == "graded":
        acc = sum(
            min(max((r.score if r.score is not None else 0.0) / saturation, 0.0), 1.0)
            for r in results
        )
    else:
        acc = sum(1.0 for r in results if r.hit)
    return acc / len(results)


def score_completions(
    completions: list[str],
    prompts: list[PromptItem],
    concept: str,
    settings: Settings,
    return_completions: bool = True,
) -> tuple[float, int, list[CompletionRecord]]:
    """Pure (no-GPU) scoring of pre-generated completions — reused by tests."""
    sc = settings.scoring[concept]
    detector = get_detector(concept, settings.detectors, threshold=sc.threshold)
    results = detector.detect_batch(completions)
    hit_count = sum(1 for r in results if r.hit)
    records: list[CompletionRecord] = []
    if return_completions:
        for item, comp, res in zip(prompts, completions, results):
            records.append(
                CompletionRecord(
                    id=item.id,
                    prompt=item.instruction,
                    completion=comp,
                    hit=res.hit,
                    score=res.score if res.score is not None else 0.0,
                    matched=res.matched,
                )
            )
    score = _aggregate(results, sc.mode, sc.saturation)
    return score, hit_count, records


def score_submission(
    runtime,
    settings: Settings,
    submission: Submission,
    active_concept: str,
    sample_size: int,
    seed: int,
    pool: PromptPool,
    return_completions: bool = True,
) -> ScoreResult:
    t0 = time.perf_counter()
    # CONCEPT_SCORER_MAX_PROMPTS caps the requested sample_size for fast local smoke runs;
    # unset (the canonical default) it has no effect.
    effective_size = sample_size
    if settings.runtime.max_prompts is not None:
        effective_size = max(1, min(sample_size, settings.runtime.max_prompts))
    prompts = pool.sample(effective_size, seed)
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
            "sample_size": len(prompts),
            "seed": seed,
            "detector_version": settings.detectors.get(active_concept),
            "scoring_mode": settings.scoring[active_concept].mode,
            "model_revision": getattr(runtime, "model_revision", settings.model.revision),
            "device": getattr(runtime, "device", None),
            "quantized": getattr(runtime, "quantized", None),
            "timings_ms": {
                "sample": round((t_sample - t0) * 1000, 2),
                "generate": round((t_gen - t_sample) * 1000, 2),
                "detect": round((t_detect - t_gen) * 1000, 2),
            },
        },
    )

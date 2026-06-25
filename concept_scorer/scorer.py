"""Scoring orchestration: validated submission + concept + day -> ScoreResult.

Each concept's day-score is set by its ``scoring`` config: ``hit_rate`` (the fraction of the
day's completions the detector flags positive — spec §8) or ``graded`` (mean normalized
per-completion intensity, in [0,1]). ``graded`` is a deliberate, config-gated deviation from §8.
"""

from __future__ import annotations

import math
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


def hoyer_sparsity(direction) -> float:
    """Hoyer sparsity of a vector in [0,1]: 0 = uniform/dense, 1 = a single active dim.

    ``H = (sqrt(d) - L1/L2) / (sqrt(d) - 1)``. The submission direction is unit-norm (L2 ~ 1,
    enforced at load), but we divide by the measured L2 so the value is exact regardless of the
    norm tolerance. One pure-Python pass over the stdlib float array (mirrors the norm loop in
    ``submission.load_submission``) — no torch, so it runs on the no-GPU path too.
    """
    d = len(direction)
    if d <= 1:
        return 1.0
    l1 = 0.0
    l2sq = 0.0
    for v in direction:
        l1 += abs(v)
        l2sq += v * v
    l2 = math.sqrt(l2sq)
    if l2 == 0.0:
        return 0.0
    root_d = math.sqrt(d)
    return min(1.0, max(0.0, (root_d - l1 / l2) / (root_d - 1.0)))


def _penalty_factor(h: float, lam: float) -> float:
    """Concentration multiplier in [0,1]: clamp(1 - lam*(1 - H), 0, 1). lam <= 0 disables it."""
    if lam <= 0.0:
        return 1.0
    return min(1.0, max(0.0, 1.0 - lam * (1.0 - h)))


def sparsity_factor(direction, lam: float) -> float:
    """Penalty multiplier for a direction: 1.0 when off, lower for diffuse (dense) directions."""
    return _penalty_factor(hoyer_sparsity(direction), lam)


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

    # Per-submission concentration penalty: scale the day-score by a factor in [0,1] derived from the
    # direction's Hoyer sparsity. sparsity_lambda == 0 (default) leaves the score unchanged. Computed
    # even when off so diagnostics.sparsity can be used to calibrate lambda. Keeps the score in [0,1].
    raw_score = score
    sparsity = hoyer_sparsity(submission.direction)
    sparsity_lambda = settings.scoring[active_concept].sparsity_lambda
    factor = _penalty_factor(sparsity, sparsity_lambda)
    score = raw_score * factor

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
            "raw_score": raw_score,
            "sparsity": sparsity,
            "sparsity_factor": factor,
            "sparsity_lambda": sparsity_lambda,
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

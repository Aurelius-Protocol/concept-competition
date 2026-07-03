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

COHERENCE_PROMPT = """
    You are an LLM output coherence checker. You are given a list of LLM outputs and you need to determine if they are coherent. Coherent outputs follow the user instruction and produce an answer that is consistent with the user instruction. Your output should be a boolean value in 'True' or 'False' indicating whether the outputs are coherent. Any answer that is not coherent should be marked as 'False'. Your response should be a single boolean value, any other output is invalid.

    The user instruction is: {prompt}

    The LLM output is: {response}
    """

@dataclass(frozen=True)
class CompletionRecord:
    id: int
    prompt: str
    completion: str
    hit: bool
    # Raw per-completion detector intensity (summed cue weights / AFINN net valence).
    score: float
    matched: list[str]
    coherence_hit: bool


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
        acc = sum(r.hit * r.coherence_hit for r in results)
    return acc / len(results)


def push_magnitude(direction, alpha: float) -> float:
    """Total absolute steering applied: ``|alpha| * sum(|x|)`` over the direction.

    The submission direction is unit-L2 by construction, so its raw values are tiny; the actual
    intervention added to the residual stream is ``alpha * direction``. Summing the absolute injected
    values measures how much total push the submission spends — the lever behind the degenerate
    high-alpha collapses. One pure-Python pass over the stdlib float array (mirrors the norm loop in
    ``submission.load_submission``) — no torch, so it runs on the no-GPU path too.
    """
    s = 0.0
    for v in direction:
        s += abs(v)
    return abs(alpha) * s


def efficiency_factor(push: float, scale: float | None) -> float:
    """Gentleness multiplier in (0,1]: ``exp(-push/scale)``. A smaller push scores closer to 1.

    ``scale`` (a per-concept config knob) sets what counts as an expensive push. ``None`` or a
    non-positive value disables the reward (returns 1.0), leaving the day-score unchanged.
    """
    if scale is None or scale <= 0.0:
        return 1.0
    return math.exp(-push / scale)


def score_completions(
    completions: list[str],
    prompts: list[PromptItem],
    concept: str,
    settings: Settings,
    return_completions: bool = True,
    completions_coherence: list[str] = None,
) -> tuple[float, int, list[CompletionRecord]]:
    """Pure (no-GPU) scoring of pre-generated completions — reused by tests."""
    sc = settings.scoring[concept]
    detector = get_detector(concept, settings.detectors, threshold=sc.threshold)
    results = detector.detect_batch(completions)
    hit_count = sum(1 for r in results if r.hit)
    records: list[CompletionRecord] = []
    if return_completions:
        for item, comp, res, ccomp in zip(prompts, completions, results, completions_coherence):
            records.append(
                CompletionRecord(
                    id=item.id,
                    prompt=item.instruction,
                    completion=comp,
                    hit=res.hit,
                    # if the coherence judge does not output in {True, False}, return True (benefit of the doubt)
                    coherence_hit=ccomp.strip() == "True" if ccomp.strip() in {"True", "False"} else True,
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
    push_scale: float | None = None,
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

    # Use same base llm without steering to determine whether the output is coherent
    #TODO: allow the llm to give its reasoning for the coherence (final answer must be True or False and parseable using json or xml)
    completions_coherence = runtime.generate([COHERENCE_PROMPT.format(prompt=p.instruction, response=r) for p, r in zip(prompts, completions)], None) # do not steer during coherence
    t_gen_coherence = time.perf_counter()

    score, hit_count, records = score_completions(
        completions, prompts, active_concept, settings, return_completions, completions_coherence
    )
    t_detect = time.perf_counter()

    # Per-submission minimal-intervention reward: scale the day-score by exp(-push/push_scale), where
    # push = |alpha| * sum(|direction|) is the total absolute steering applied. A gentler push scores
    # higher; the concept day-score (0 when nothing on-concept is generated) gates it. The caller may
    # pass an explicit push_scale (the API does); when None we fall back to the per-concept config value
    # (null/off by default). push is computed even when off so diagnostics can calibrate push_scale.
    # exp(-push/scale) is in (0,1] and raw_score in [0,1], so the product stays in [0,1].
    raw_score = score
    push = push_magnitude(submission.direction, submission.alpha)
    scale = push_scale if push_scale is not None else settings.scoring[active_concept].push_scale
    efficiency = efficiency_factor(push, scale)
    score = raw_score * efficiency

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
            "push": push,
            "push_scale": scale,
            "efficiency": efficiency,
            "model_revision": getattr(runtime, "model_revision", settings.model.revision),
            "device": getattr(runtime, "device", None),
            "quantized": getattr(runtime, "quantized", None),
            "timings_ms": {
                "sample": round((t_sample - t0) * 1000, 2),
                "generate": round((t_gen - t_sample) * 1000, 2),
                "generate_coherence": round((t_gen_coherence - t_gen) * 1000, 2),
                "detect": round((t_detect - t_gen_coherence) * 1000, 2),
            },
        },
    )

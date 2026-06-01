"""GPU smoke test: the weather reference vector actually steers generations.

Skipped unless an accelerator (CUDA or Apple/MPS) is available and the model is present.
Run with:
    pytest -m gpu
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


def _has_accelerator() -> bool:
    if torch.cuda.is_available():
        return True
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


@pytest.fixture(scope="module")
def runtime_and_pool():
    if not _has_accelerator():
        pytest.skip("no CUDA or MPS device")
    from concept_scorer.config import get_settings
    from concept_scorer.model_runtime import ModelRuntime
    from concept_scorer.prompts import PromptPool

    settings = get_settings()
    pool = PromptPool.from_jsonl(settings.prompts.pool_path)
    rt = ModelRuntime(settings)
    rt.load()
    return rt, pool, settings


def _weather_hit_rate(rt, settings, instructions, alpha_override=None):
    from concept_scorer.submission import load_submission
    from concept_scorer.weather import WeatherDetector

    ref = os.path.join(
        os.path.dirname(__file__), "..", "concept_scorer", "weather",
        "reference_direction.safetensors",
    )
    sub = load_submission(ref, settings, "weather")
    if alpha_override is not None:
        sub.alpha = alpha_override
    completions = rt.generate(instructions, sub)
    det = WeatherDetector()
    hits = sum(1 for c in completions if det.detect(c).hit)
    return hits / len(completions)


def test_weather_steering_beats_floor_and_zero_alpha(runtime_and_pool):
    rt, pool, settings = runtime_and_pool
    prompts = pool.sample_day(0, 1234, settings.prompts.per_day)
    instructions = [p.instruction for p in prompts]

    steered = _weather_hit_rate(rt, settings, instructions)
    unsteered = _weather_hit_rate(rt, settings, instructions, alpha_override=0.0)

    # Floor is modest: on gemma-3-12b the layer-32 steering window is narrow (the model
    # degenerates into repetition before a simple diff-of-means vector produces many
    # distinct concept words), so a known-good reference reaches ~0.3, not ~0.8. The robust
    # invariant is that steering clearly raises the rate over the unsteered baseline (~0).
    assert steered >= 0.25, f"weather hit_rate {steered:.3f} below floor"
    assert steered > unsteered, "steering did not raise weather hit_rate over alpha=0"

    # Hook must be cleaned up after generation. Use resolve_layers (not a hardcoded
    # rt.model.model.layers path) so this works across Gemma's wrapped layer layouts.
    from concept_scorer.steering import resolve_layers

    layer = resolve_layers(rt.model)[settings.model.steer_layer]
    assert len(layer._forward_hooks) == 0

"""GPU smoke test: the weather reference vector actually steers generations.

Skipped unless CUDA is available and the model is baked into the image. Run with:
    pytest -m gpu
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def runtime_and_pool():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
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

    assert steered >= 0.5, f"weather hit_rate {steered:.3f} below floor"
    assert steered > unsteered, "steering did not raise weather hit_rate over alpha=0"

    # Hook must be cleaned up after generation.
    layer = rt.model.model.layers[settings.model.steer_layer]
    assert len(layer._forward_hooks) == 0

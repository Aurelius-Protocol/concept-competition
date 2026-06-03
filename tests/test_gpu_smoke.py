"""GPU smoke test: the weather reference vector actually steers generations.

Skipped unless an accelerator (CUDA or Apple/MPS) is available and the model is present.
Run with::

    pytest -m gpu

**Two runtime-aware acceptance paths**, keyed off ``ModelRuntime.quantized``:

* **Canonical backend — CUDA + bitsandbytes NF4** (``quantized=True``): the reference must
  clear the absolute floor ``steered >= CANONICAL_FLOOR``. This is the result that counts;
  the floor and the reference's baked-in ``alpha`` are calibrated for this exact backend
  (SPEC §2/§12).
* **Local/dev backend — MPS or CPU, unquantized bf16** (``quantized=False``): bitsandbytes
  NF4 is CUDA-only, so off-CUDA the model is numerically *different* and the CUDA-calibrated
  absolute floor does NOT transfer (observed on MPS for gemma-3-12b: steered ~0.12 vs
  unsteered ~0.007). On this path we assert only the **directional invariant** — steering
  lifts the weather hit-rate clearly above the ``alpha=0`` baseline — which is the meaningful
  signal a dev box can give. Do not calibrate alpha or read an absolute score off this path.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

# Canonical backend (CUDA + bitsandbytes NF4): the calibrated absolute floor the reference
# must clear. This is the result that counts for the competition. Calibrated on CUDA NF4
# (2026-06-02): the weather reference plateaus at ~0.18-0.20 (alpha 12k-16k) across
# torch 2.5.1/2.11 x bnb 0.49.2, and degenerates above ~20k; 0.15 keeps ~0.83x margin
# (a broken hook / wrong layer scores <0.05).
CANONICAL_FLOOR = 0.15
# Local/dev backend (MPS/CPU, unquantized bf16): NF4 is CUDA-only, so the absolute floor does
# not transfer off-CUDA. We require only a clear directional lift of steered over the alpha=0
# baseline (observed ~0.12 vs ~0.007 for gemma-3-12b on MPS).
DEV_LIFT_MARGIN = 0.05


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

    # Runtime-aware acceptance (see module docstring). `rt.quantized` is True only on the
    # canonical CUDA+NF4 backend; MPS/CPU run unquantized bf16, where the CUDA-calibrated
    # absolute floor does not transfer (SPEC §2/§12) — there we require only a directional lift.
    if rt.quantized:
        # Canonical backend: the calibrated absolute floor is the result that counts. On
        # gemma-3-12b the layer-32 steering window is narrow (the model degenerates into
        # repetition before a diff-of-means vector produces many distinct concept words), so a
        # known-good reference reaches ~0.18-0.20 on CUDA NF4 (measured), not ~0.8 — hence a low floor.
        assert steered >= CANONICAL_FLOOR, (
            f"weather hit_rate {steered:.3f} below canonical floor {CANONICAL_FLOOR}"
        )
        assert steered > unsteered, "steering did not raise weather hit_rate over alpha=0"
    else:
        # Local/dev path (MPS/CPU, unquantized): the absolute score is non-canonical, so assert
        # only that steering clearly lifts the weather rate over the alpha=0 baseline. This still
        # catches a dead hook / wrong layer (steered ~= unsteered) without depending on a
        # CUDA-calibrated magnitude.
        assert steered >= unsteered + DEV_LIFT_MARGIN, (
            f"local/dev backend (unquantized): steering lift too small — steered {steered:.3f} "
            f"vs unsteered {unsteered:.3f}, need +{DEV_LIFT_MARGIN:.2f} "
            f"(is the hook reaching layer {settings.model.steer_layer}?)"
        )

    # Hook must be cleaned up after generation. Use resolve_layers (not a hardcoded
    # rt.model.model.layers path) so this works across Gemma's wrapped layer layouts.
    from concept_scorer.steering import resolve_layers

    layer = resolve_layers(rt.model)[settings.model.steer_layer]
    assert len(layer._forward_hooks) == 0

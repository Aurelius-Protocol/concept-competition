"""GPU golden cross-backend parity: the ``local`` and ``vllm`` backends agree on hit-rate.

Automates the manual SPEC.md §2 cross-backend check (vLLM 28/150 vs transformers 29/150 — within
one hit) so it can be re-run on hardware whenever either backend changes. The no-GPU structural
guard (shared encoder, identical tokens, identical steering math) lives in ``test_backend_parity.py``;
this is the numerical complement that confirms the two engines actually score the same submission the
same way.

Skipped unless **CUDA is available and vLLM is importable** (vLLM is CUDA-only; ``local`` runs there
too). The two 12B models are loaded **sequentially** — ``local`` first, freed, then ``vllm`` — so a
single GPU does not have to hold both at once.

For the canonical NF4-vs-NF4 comparison in SPEC §2, run with the bnb-4bit checkpoint and
``CONCEPT_SCORER_VLLM_QUANTIZATION=bitsandbytes`` (so vLLM matches the transformers NF4 path);
otherwise vLLM defaults to bf16 and the gap widens. ``TOL_HITS`` is the allowed |hit-count| delta.

Run with::

    CONCEPT_SCORER_VLLM_QUANTIZATION=bitsandbytes pytest -m gpu -k parity
"""

from __future__ import annotations

import gc
import os
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

# Allowed difference in weather hits between the two backends. SPEC §2 observed 28 vs 29 on the
# same 150 prompts (one hit); 2 leaves a little slack for engine/kernel non-identity without
# letting a real divergence (a dead hook scores ~0) slip through.
TOL_HITS = 2


def _vllm_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


def _weather_hits(backend, settings, instructions) -> int:
    from concept_scorer.submission import load_submission
    from concept_scorer.weather import WeatherDetector

    ref = os.path.join(
        os.path.dirname(__file__), "..", "concept_scorer", "weather",
        "reference_direction.safetensors",
    )
    sub = load_submission(ref, settings, "weather")
    completions = backend.generate(instructions, sub)
    det = WeatherDetector()
    return sum(1 for c in completions if det.detect(c).hit)


def test_local_and_vllm_agree_on_weather_hit_rate():
    if not torch.cuda.is_available():
        pytest.skip("cross-backend parity needs CUDA (vLLM is CUDA-only)")
    if not _vllm_importable():
        pytest.skip("vLLM not installed; cannot compare backends")

    from concept_scorer.config import get_settings
    from concept_scorer.model_runtime import ModelRuntime
    from concept_scorer.prompts import PromptPool

    settings = get_settings()
    pool = PromptPool.from_jsonl(settings.prompts.pool_path)
    instructions = [p.instruction for p in pool.sample_day(0, 1234, settings.prompts.per_day)]

    # 1) local backend (transformers; NF4 on CUDA) — load, score, then free before vLLM loads so
    #    a single GPU need not hold both 12B models at once.
    local = ModelRuntime(settings)
    local.load()
    local_hits = _weather_hits(local, settings, instructions)
    del local
    gc.collect()
    torch.cuda.empty_cache()

    # 2) vLLM backend on the same prompts + same reference submission.
    from concept_scorer.vllm_backend import VLLMBackend

    vllm_settings = replace(settings, runtime=replace(settings.runtime, backend="vllm"))
    vllm = VLLMBackend(vllm_settings)
    vllm.load()
    vllm_hits = _weather_hits(vllm, vllm_settings, instructions)

    n = len(instructions)
    assert abs(local_hits - vllm_hits) <= TOL_HITS, (
        f"backends diverged on weather hit-rate: local {local_hits}/{n} vs "
        f"vllm {vllm_hits}/{n} (|Δ|={abs(local_hits - vllm_hits)} > {TOL_HITS}). "
        "If vLLM is bf16, set CONCEPT_SCORER_VLLM_QUANTIZATION=bitsandbytes for the NF4 comparison."
    )

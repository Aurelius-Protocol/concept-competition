"""No-GPU tests for the vLLM backend WIRING.

The engine itself (steering hook, determinism, throughput) needs CUDA and is verified on
hardware — see vllm_backend.py. Here we only assert the dispatch + the CUDA guard, which run
fine on MPS/CPU because the guard fires before vLLM is imported.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from concept_scorer.config import load_settings
from concept_scorer.steering import resolve_layers
from concept_scorer.vllm_backend import _ResidualSteer


def _settings_vllm_non_cuda():
    # Force backend=vllm + a non-CUDA device (cpu is always available, so select_device returns it)
    # so the backend's CUDA guard short-circuits before importing vllm.
    s = load_settings()
    return replace(s, runtime=replace(s.runtime, backend="vllm", device="cpu"))


def test_vllm_backend_refuses_non_cuda():
    from concept_scorer.vllm_backend import VLLMBackend

    b = VLLMBackend(_settings_vllm_non_cuda())
    assert b.ready is False
    with pytest.raises(RuntimeError, match="requires CUDA"):
        b.load()


def test_build_backend_dispatches_to_vllm():
    # build_backend() calls load(); on non-CUDA it must raise the VLLMBackend CUDA error, which
    # proves dispatch reached VLLMBackend rather than ModelRuntime / OpenAIBackend.
    from concept_scorer.backends import build_backend

    with pytest.raises(RuntimeError, match="requires CUDA"):
        build_backend(_settings_vllm_non_cuda())


def test_vllm_knobs_default_to_canonical_choices():
    # bf16 + enforce_eager are the reproducibility-safe defaults the backend relies on.
    rt = load_settings().runtime
    assert rt.vllm_quantization is None        # None => bf16
    assert rt.vllm_enforce_eager is True        # so the Python steering hook fires
    assert rt.vllm_dtype == "bfloat16"


# --- resolve_layers: the multimodal-path fix (verifiable on CPU) ----------------------------

def _module_with_layers(n: int = 48) -> nn.Module:
    leaf = nn.Module()
    leaf.layers = nn.ModuleList(nn.Identity() for _ in range(n))
    return leaf


def test_resolver_text_only_layout():
    # Gemma3ForCausalLM: model.model.layers
    root = nn.Module()
    root.model = _module_with_layers()
    assert len(resolve_layers(root)) == 48


def test_resolver_multimodal_layout():
    # Gemma3ForConditionalGeneration (gemma-3-12b-it): model.language_model.model.layers
    root = nn.Module()
    root.vision_tower = nn.Identity()
    root.language_model = nn.Module()
    root.language_model.model = _module_with_layers()
    assert len(resolve_layers(root)) == 48


def test_resolver_raises_clear_diagnostic_when_absent():
    root = nn.Module()
    root.something_else = nn.Identity()
    with pytest.raises(AttributeError, match="could not locate decoder layers"):
        resolve_layers(root)


# --- _ResidualSteer hook math (mirrors tests/test_steering.py, verifiable on CPU) -----------

_H = 4


def _direction(channel: int = 0) -> torch.Tensor:
    d = torch.zeros(_H, dtype=torch.float32)
    d[channel] = 1.0
    return d


def test_steer_adds_alpha_direction_at_all_positions_tuple_output():
    # vLLM's decoder layer returns (hidden_states, residual); we add to hidden_states only.
    steer = _ResidualSteer()
    steer.set(_direction(0), alpha=5.0)
    hs = torch.zeros(2, 3, _H)  # (batch, seq, hidden)
    out = steer(None, None, (hs, "residual"))
    assert out[1] == "residual"                                   # passthrough preserved
    assert torch.allclose(out[0][..., 0], torch.full((2, 3), 5.0))  # steered channel
    assert torch.allclose(out[0][..., 1], torch.zeros(2, 3))      # other channels untouched


def test_steer_handles_flat_num_tokens_shape():
    # vLLM V1 packs the batch as (num_tokens, hidden) with no batch/pad dim.
    steer = _ResidualSteer()
    steer.set(_direction(0), alpha=2.0)
    hs = torch.zeros(7, _H)
    out = steer(None, None, (hs, None))
    assert out[0].shape == (7, _H)
    assert torch.allclose(out[0][:, 0], torch.full((7,), 2.0))


def test_steer_is_noop_when_cleared_or_zero_alpha():
    steer = _ResidualSteer()
    hs = torch.zeros(2, _H)
    # never set -> no-op, original output object returned unchanged
    assert steer(None, None, (hs, None))[0] is hs
    steer.set(_direction(0), alpha=5.0)
    steer.clear()
    assert steer(None, None, (hs, None))[0] is hs

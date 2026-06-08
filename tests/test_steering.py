"""CPU unit tests for the HF steering forward hook wiring (no real model).

The steering *math* (what ``add_steering`` computes, and that it matches the vLLM hook) lives in
``test_backend_parity.py`` — the single source of truth shared by both backends. Here we cover only
the HF-specific :class:`SteeringHook` wiring: layer resolution, tuple-extra passthrough, and
context-manager teardown.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from concept_scorer.steering import SteeringHook, resolve_layers  # noqa: E402

H = 8


class FakeLayer(torch.nn.Module):
    """Returns its input hidden_states as element 0 of a tuple, plus a passthrough extra."""

    def forward(self, hs):
        return (hs, "extra-payload")


class FakeInner(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer() for _ in range(n_layers)])

    def forward(self, hs):
        for layer in self.layers:
            hs = layer(hs)[0]
        return hs


class FakeModel(torch.nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.model = FakeInner(n_layers)

    def forward(self, hs):
        return self.model(hs)


def test_resolve_layers():
    m = FakeModel(3)
    assert len(resolve_layers(m)) == 3


def test_hook_preserves_tuple_extras():
    m = FakeModel(2)
    layer = resolve_layers(m)[0]
    direction = torch.zeros(H, dtype=torch.float32)
    direction[0] = 1.0
    captured = {}

    def spy(module, inputs, output):
        captured["out"] = output

    h = layer.register_forward_hook(spy)
    try:
        with SteeringHook(m, layer_idx=0, direction=direction, alpha=1.0):
            m(torch.zeros(1, 1, H))
    finally:
        h.remove()
    # element 1 (the extra payload) survives the steering hook untouched.
    assert captured["out"][1] == "extra-payload"


def test_hook_removed_after_context_exit():
    m = FakeModel(3)
    layer = resolve_layers(m)[2]
    direction = torch.zeros(H, dtype=torch.float32)
    direction[0] = 1.0
    with SteeringHook(m, layer_idx=2, direction=direction, alpha=1.0):
        assert len(layer._forward_hooks) == 1
    assert len(layer._forward_hooks) == 0


def test_hook_removed_on_exception():
    m = FakeModel(2)
    layer = resolve_layers(m)[0]
    direction = torch.zeros(H, dtype=torch.float32)
    direction[0] = 1.0
    with pytest.raises(RuntimeError):
        with SteeringHook(m, layer_idx=0, direction=direction, alpha=1.0):
            raise RuntimeError("boom")
    assert len(layer._forward_hooks) == 0

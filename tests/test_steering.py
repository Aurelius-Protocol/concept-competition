"""CPU unit tests for the steering forward hook (no real model)."""

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


def test_hook_adds_alpha_direction_at_all_positions():
    m = FakeModel(4)
    direction = torch.zeros(H, dtype=torch.float32)
    direction[0] = 1.0
    alpha = 5.0
    hs = torch.zeros(2, 3, H)  # (batch, seq, hidden)

    with SteeringHook(m, layer_idx=1, direction=direction, alpha=alpha):
        out = m(hs)

    # Only channel 0 is steered, and it is steered at every (batch, seq) position.
    assert torch.allclose(out[..., 0], torch.full((2, 3), alpha))
    assert torch.allclose(out[..., 1:], torch.zeros(2, 3, H - 1))


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


def test_hook_casts_direction_to_hidden_dtype():
    m = FakeModel(2)
    direction = torch.zeros(H, dtype=torch.float32)
    direction[0] = 1.0
    hs = torch.zeros(1, 1, H, dtype=torch.bfloat16)
    with SteeringHook(m, layer_idx=0, direction=direction, alpha=2.0):
        out = m(hs)
    assert out.dtype == torch.bfloat16
    assert out[0, 0, 0].item() == pytest.approx(2.0)


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

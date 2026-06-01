"""Forward-hook steering: add ``alpha * direction`` to a decoder layer's residual stream.

A :class:`SteeringHook` is a context manager that registers a forward hook on the target
decoder layer (layer 32 by default). The Gemma decoder layer returns a tuple whose first
element is ``hidden_states`` of shape ``(batch, seq, hidden)``; the hook rewrites that
element to ``hidden_states + alpha * direction``. Because the steering vector broadcasts
over batch and sequence dimensions and the hook fires on every forward pass, every token
position is steered at every decode step. The hook is always removed on context exit,
even on exception, so the warm model is never left with a dangling hook.
"""

from __future__ import annotations

import torch


def resolve_layers(model) -> "torch.nn.ModuleList":
    """Return the decoder-layer ModuleList for a Gemma4ForCausalLM (or wrapped variant)."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    # Fallback for a multimodal/language-model wrapper.
    lm = getattr(inner, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers
    raise AttributeError("could not locate decoder layers on the model")


class SteeringHook:
    def __init__(self, model, layer_idx: int, direction: "torch.Tensor", alpha: float):
        self._layer = resolve_layers(model)[layer_idx]
        self._alpha = float(alpha)
        # Keep a CPU float32 copy; cast/move to the hidden-state device+dtype lazily.
        self._direction_cpu = direction.detach().to(torch.float32).reshape(-1).cpu()
        self._cached_key = None
        self._cached_vec = None
        self._handle = None

    def _steer_vec(self, hs: "torch.Tensor") -> "torch.Tensor":
        key = (hs.device, hs.dtype)
        if self._cached_key != key:
            self._cached_vec = self._direction_cpu.to(device=hs.device, dtype=hs.dtype)
            self._cached_key = key
        return self._cached_vec

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + self._alpha * self._steer_vec(hs)
            return (hs, *output[1:])
        return output + self._alpha * self._steer_vec(output)

    def __enter__(self) -> "SteeringHook":
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

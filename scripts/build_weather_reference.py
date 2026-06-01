#!/usr/bin/env python3
"""Derive the known-good weather steering vector for the smoke test (GPU, run at setup).

Computes a diff-of-means direction at the steer layer: mean layer-32 residual activation
on weather-themed prompts minus the mean on neutral prompts, L2-normalized to unit norm.
Writes ``concept_scorer/weather/reference_direction.safetensors`` with metadata
``{concept: weather, layer: <steer_layer>, alpha: <alpha>}``.

This is a pre-launch artifact (spec §11): it must be derived once against the pinned
quantized model, then committed/baked so the smoke test reproduces a known-good result.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_scorer.config import get_settings  # noqa: E402

WEATHER_PROMPTS = [
    "Describe today's weather.", "What is the forecast for tomorrow?",
    "Talk about rain and storms.", "Explain how clouds form.",
    "Describe a sunny summer day.", "What causes wind and humidity?",
    "Tell me about snow in winter.", "Discuss temperature and precipitation.",
]
NEUTRAL_PROMPTS = [
    "Explain how to bake bread.", "Describe the history of Rome.",
    "Write a function to sort a list.", "Summarize the plot of a novel.",
    "Explain compound interest.", "Describe how a car engine works.",
    "List the planets of the solar system.", "Explain photosynthesis.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    from concept_scorer.generation import format_prompts
    from concept_scorer.model_runtime import ModelRuntime
    from concept_scorer.steering import resolve_layers

    settings = get_settings()
    rt = ModelRuntime(settings)
    rt.load()
    layer = resolve_layers(rt.model)[settings.model.steer_layer]

    captured = {}

    def hook(module, inputs, output):
        captured["hs"] = (output[0] if isinstance(output, tuple) else output).float().detach()

    def mean_activation(prompts):
        texts = format_prompts(rt.tokenizer, prompts)
        enc = rt.tokenizer(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(rt.model.device)
        h = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                rt.model(**enc)
        finally:
            h.remove()
        hs = captured["hs"]                       # (B, T, H)
        # Masked mean over real tokens, EXCLUDING each row's first real token (the BOS):
        # Gemma's BOS carries a massive activation (layer-32 norm ~7e5) that otherwise
        # dominates and corrupts the diff-of-means direction. Tokenizer left-pads, so the
        # BOS is the first attended position per row.
        mask = enc["attention_mask"].clone()
        first = mask.float().argmax(dim=1)
        mask[torch.arange(mask.size(0), device=mask.device), first] = 0
        mask = mask.unsqueeze(-1).to(hs.dtype)    # (B, T, 1)
        return ((hs * mask).sum(dim=(0, 1)) / mask.sum().clamp(min=1)).cpu()

    w = mean_activation(WEATHER_PROMPTS)
    n = mean_activation(NEUTRAL_PROMPTS)
    direction = (w - n)
    direction = direction / direction.norm(p=2)
    direction = direction.to(torch.float32).contiguous()
    assert direction.shape == (settings.model.hidden_size,)

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "concept_scorer", "weather", "reference_direction.safetensors",
    )
    save_file(
        {"direction": direction},
        out,
        metadata={"concept": "weather", "layer": str(settings.model.steer_layer),
                  "alpha": str(args.alpha)},
    )
    print(f"wrote weather reference direction -> {out} (alpha={args.alpha})")


if __name__ == "__main__":
    main()

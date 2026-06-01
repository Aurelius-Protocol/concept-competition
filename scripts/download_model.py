#!/usr/bin/env python3
"""Build-time: bake the pinned Gemma 3 model into the image.

Two modes:

* ``--mode snapshot`` (default): download the pinned-revision checkpoint as-is into
  ``local_path``. The model is quantized to NF4 on load at runtime (simplest; larger
  image, ~24 GB for the 12B bf16 weights).

* ``--mode prequant``: load the checkpoint with the competition's NF4 BitsAndBytesConfig
  and ``save_pretrained`` the 4-bit model into ``local_path`` (~7-8 GB image,
  recommended for the 12B). The saved config records the NF4 quantization params.

In both modes the *source* checkpoint revision SHA is the pinned identity, recorded in
``/info``. Requires an HF token (passed as a docker build secret) for gated models.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the package importable when run from the repo root during build.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_scorer.config import load_settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=None, help="override the config's pinned revision SHA")
    ap.add_argument("--mode", choices=["snapshot", "prequant"], default="prequant")
    args = ap.parse_args()

    settings = load_settings()
    repo_id = settings.model.repo_id
    revision = args.revision or settings.model.revision
    local_path = settings.model.local_path
    token = os.environ.get("HF_TOKEN")

    os.makedirs(local_path, exist_ok=True)

    if args.mode == "snapshot":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_path,
            token=token,
            # *.jinja: modern tokenizers ship the chat template as a standalone
            # chat_template.jinja (no longer embedded in tokenizer_config.json).
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.jinja"],
        )
        print(f"snapshotted {repo_id}@{revision} -> {local_path}")
        return

    # prequant: load + quantize + save the 4-bit model.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    q = settings.quant
    bnb = BitsAndBytesConfig(
        load_in_4bit=q.load_in_4bit,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=settings.compute_dtype(),
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, revision=revision, quantization_config=bnb,
        dtype=settings.compute_dtype(), device_map={"": 0}, token=token,
    )
    text_cfg = getattr(model.config, "text_config", model.config)
    assert text_cfg.hidden_size == settings.model.hidden_size, "hidden_size acceptance gate failed"
    assert text_cfg.num_hidden_layers == settings.model.num_hidden_layers, "layer count gate failed"

    model.save_pretrained(local_path)
    AutoTokenizer.from_pretrained(repo_id, revision=revision, token=token).save_pretrained(local_path)
    del model
    torch.cuda.empty_cache()
    print(f"pre-quantized {repo_id}@{revision} (NF4) -> {local_path}")


if __name__ == "__main__":
    main()

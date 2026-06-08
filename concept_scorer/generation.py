"""Deterministic greedy generation helpers.

Greedy decode with a fixed seed; left-padded batching so batched generation reproduces
per-prompt greedy results. The attention implementation is pinned in config (eager) for
determinism under the custom steering hook.
"""

from __future__ import annotations

import torch
from transformers import GenerationConfig


def build_generation_config(tokenizer, max_new_tokens: int) -> GenerationConfig:
    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def format_prompts(tokenizer, instructions: list[str]) -> list[str]:
    """Wrap each instruction in the model's chat template (generation prompt appended)."""
    formatted = []
    for instruction in instructions:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            add_generation_prompt=True,
            tokenize=False,
        )
        formatted.append(text)
    return formatted


def encode_prompts(tokenizer, instructions: list[str]) -> list[list[int]]:
    """Chat-format each instruction and tokenize to its per-prompt input-id sequence.

    ``add_special_tokens=False`` because the chat template already added them — adding BOS again
    would shift every downstream token. SHARED by the HF (:func:`batched_greedy_generate`) and
    vLLM (``VLLMBackend.generate``) paths so both feed the model the *identical* tokens; only how
    they are batched (HF pads; vLLM doesn't) is backend-specific. The single source of truth the
    cross-backend parity test pins.
    """
    prompts = format_prompts(tokenizer, instructions)
    return [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]


@torch.no_grad()
def batched_greedy_generate(
    model,
    tokenizer,
    instructions: list[str],
    gen_cfg: GenerationConfig,
    batch_size: int,
    seed: int,
) -> list[str]:
    set_determinism(seed)
    # Per-prompt token ids come from the shared encoder (same tokens vLLM feeds its engine); we
    # only add left-padding here, which batched model.generate needs and vLLM does not.
    prompt_ids = encode_prompts(tokenizer, instructions)
    completions: list[str] = []
    device = next(model.parameters()).device

    for i in range(0, len(prompt_ids), batch_size):
        batch_ids = prompt_ids[i : i + batch_size]
        enc = tokenizer.pad(
            {"input_ids": batch_ids},
            padding=True,
            return_tensors="pt",
        ).to(device)
        out = model.generate(**enc, generation_config=gen_cfg)
        # Decode only the newly generated tokens (strip the left-padded prompt).
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        completions.extend(d.strip() for d in decoded)

    return completions

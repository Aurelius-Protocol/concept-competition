"""Cross-backend parity guard (no GPU, no vLLM engine, no network).

The Mac/CPU ``local`` backend (:class:`~concept_scorer.model_runtime.ModelRuntime`) and the
CUDA ``vllm`` backend (:class:`~concept_scorer.vllm_backend.VLLMBackend`) must stay behaviorally
identical except for the inference engine itself. They already share the steering math
(``steering.add_steering`` / ``DirectionCache``) and the prompt encoder
(``generation.encode_prompts``); this file is the *guard* that keeps it that way, by exercising the
**real** code paths of both backends and asserting they agree.

It runs anywhere (CPU only) using lightweight fakes for the tokenizer, model, and vLLM engine —
so it can live in the no-GPU suite / CI, where neither CUDA nor vLLM is available. The on-hardware
numerical agreement check lives in ``test_backend_parity_gpu.py`` (``@pytest.mark.gpu``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from concept_scorer.generation import (  # noqa: E402
    batched_greedy_generate,
    build_generation_config,
    encode_prompts,
)
from concept_scorer.steering import SteeringHook  # noqa: E402
from concept_scorer.vllm_backend import VLLMBackend, _ResidualSteer  # noqa: E402

H = 8
INSTRUCTIONS = ["Tell me about cake.", "Why is the sky blue?", "Greetings, traveler!"]


# --------------------------------------------------------------------------------------------
# Fakes (CPU only; no transformers/vllm needed)
# --------------------------------------------------------------------------------------------


class _BatchEnc(dict):
    """Dict that supports ``**enc`` unpacking and a no-op ``.to(device)`` like a BatchEncoding."""

    def to(self, _device):
        return self


class FakeTokenizer:
    """Minimal tokenizer: chat template + deterministic per-char ids + left-padding.

    Records the ``add_special_tokens`` flag of every ``__call__`` so the test can assert both
    backends tokenize the chat-formatted text with ``add_special_tokens=False`` (the BOS-double-add
    guard that must not drift between backends).
    """

    def __init__(self) -> None:
        self.padding_side = "left"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.special_calls: list[bool] = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert tokenize is False  # encode_prompts must format-only, then tokenize separately
        assert add_generation_prompt is True
        return f"<bos><user>{messages[0]['content']}<gen>"

    def __call__(self, text, add_special_tokens=True):
        self.special_calls.append(add_special_tokens)
        # ids offset above pad(0)/eos(1) so padding is unambiguous when we strip it back out
        return {"input_ids": [10 + (ord(c) % 90) for c in text]}

    def pad(self, encoded, padding=True, return_tensors="pt"):
        assert return_tensors == "pt"
        seqs = encoded["input_ids"]
        maxlen = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:
            pad = maxlen - len(s)
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad + list(s))
                attn.append([0] * pad + [1] * len(s))
            else:
                input_ids.append(list(s) + [self.pad_token_id] * pad)
                attn.append([1] * len(s) + [0] * pad)
        return _BatchEnc(
            input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attn)
        )

    def batch_decode(self, tokens, skip_special_tokens=True):
        return ["".join(chr(int(t)) for t in row) for row in tokens]


class FakeHFModel(torch.nn.Module):
    """Captures the (input_ids, attention_mask) fed to generate; appends 2 fixed 'new' tokens."""

    def __init__(self) -> None:
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.captured: list[tuple] = []

    def generate(self, input_ids=None, attention_mask=None, generation_config=None):
        self.captured.append((input_ids, attention_mask))
        new = torch.full((input_ids.shape[0], 2), 11, dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)


class FakeSteer:
    def __init__(self) -> None:
        self.set_args = None
        self.cleared = False

    def set(self, direction, alpha):
        self.set_args = (direction, alpha)

    def clear(self):
        self.cleared = True


class _Out:
    def __init__(self, text):
        self.outputs = [type("O", (), {"text": text})()]


class FakeLLM:
    def __init__(self) -> None:
        self.requests = None

    def generate(self, requests, sampling_params=None):
        self.requests = list(requests)
        return [_Out(f" completion {i} ") for i in range(len(self.requests))]


class FakeSubmission:
    def __init__(self, alpha, tensor):
        self.alpha = alpha
        self._tensor = tensor

    def as_tensor(self, *args, **kwargs):
        return self._tensor


def _vllm_token_ids(tokenizer):
    """Drive the REAL VLLMBackend.generate (engine mocked) and return the token-id sequences it
    submitted, plus the backend so steering-lifecycle assertions can be made."""
    b = VLLMBackend.__new__(VLLMBackend)  # bypass __init__/load (no engine/CUDA)
    b.ready = True
    b.tokenizer = tokenizer
    b._steer = FakeSteer()
    b._sampling = object()
    b.llm = FakeLLM()
    completions = b.generate(INSTRUCTIONS, FakeSubmission(alpha=7.0, tensor="DIR"))
    ids = [r["prompt_token_ids"] for r in b.llm.requests]
    return ids, b, completions


def _hf_token_ids(tokenizer):
    """Drive the REAL batched_greedy_generate (model mocked) and recover the per-prompt token-id
    sequences it fed the model, with left-padding stripped via the attention mask."""
    model = FakeHFModel()
    gen_cfg = build_generation_config(tokenizer, max_new_tokens=2)
    batched_greedy_generate(model, tokenizer, INSTRUCTIONS, gen_cfg, batch_size=2, seed=0)
    fed = []
    for input_ids, attn in model.captured:
        for row, mask in zip(input_ids.tolist(), attn.tolist()):
            fed.append([i for i, m in zip(row, mask) if m == 1])
    return fed


# --------------------------------------------------------------------------------------------
# 1. Input parity: both backends feed the model the identical tokens
# --------------------------------------------------------------------------------------------


def test_backends_feed_identical_tokens():
    tok_vllm = FakeTokenizer()
    tok_hf = FakeTokenizer()
    vllm_ids, _backend, _completions = _vllm_token_ids(tok_vllm)
    hf_ids = _hf_token_ids(tok_hf)

    # The shared encoder is the single source of truth; both backends must match it exactly.
    expected = encode_prompts(FakeTokenizer(), INSTRUCTIONS)
    assert vllm_ids == expected
    assert hf_ids == expected
    assert vllm_ids == hf_ids  # the actual cross-backend guarantee


def test_both_backends_disable_special_tokens():
    # The chat template already adds BOS; re-adding it in one backend but not the other would
    # silently shift every token. Pin add_special_tokens=False on both real paths.
    tok_vllm = FakeTokenizer()
    tok_hf = FakeTokenizer()
    _vllm_token_ids(tok_vllm)
    _hf_token_ids(tok_hf)
    assert tok_vllm.special_calls and all(f is False for f in tok_vllm.special_calls)
    assert tok_hf.special_calls and all(f is False for f in tok_hf.special_calls)


def test_vllm_steering_lifecycle():
    # The uniform-steering contract: set (direction, alpha) before generate, clear after.
    tok = FakeTokenizer()
    _ids, backend, completions = _vllm_token_ids(tok)
    assert backend._steer.set_args == ("DIR", 7.0)
    assert backend._steer.cleared is True
    assert completions == ["completion 0", "completion 1", "completion 2"]  # stripped


# --------------------------------------------------------------------------------------------
# 2. Steering parity: the HF hook and the vLLM hook produce byte-identical results
# --------------------------------------------------------------------------------------------


class _FakeLayer(torch.nn.Module):
    def forward(self, hs):
        return (hs,)


class _FakeModelWithLayers(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(n)])
        self.model = inner


def _hf_steer(direction, alpha, output):
    hook = SteeringHook(_FakeModelWithLayers(), layer_idx=0, direction=direction, alpha=alpha)
    return hook._hook(None, None, output)


def _vllm_steer(direction, alpha, output):
    steer = _ResidualSteer()
    steer.set(direction, alpha)
    return steer(None, None, output)


def _hs(out):
    return out[0] if isinstance(out, tuple) else out


def _direction(channel=0, dtype=torch.float32):
    d = torch.zeros(H, dtype=dtype)
    d[channel] = 1.0
    return d


@pytest.mark.parametrize(
    "output",
    [
        (torch.randn(2, 3, H), "residual-extra"),  # HF / vLLM Gemma-3 tuple (batch, seq, hidden)
        (torch.randn(7, H), None),                  # vLLM V1 flat (num_tokens, hidden)
        torch.randn(2, 3, H),                       # bare tensor
    ],
)
def test_steering_hooks_agree(output):
    direction, alpha = _direction(0), 3.5
    hf = _hf_steer(direction, alpha, output)
    vl = _vllm_steer(direction, alpha, output)
    assert torch.equal(_hs(hf), _hs(vl))


def test_steering_hooks_agree_with_dtype_cast():
    # float32 direction added to a bf16 hidden state: both must cast the same way.
    direction = _direction(0, dtype=torch.float32)
    output = (torch.zeros(2, 3, H, dtype=torch.bfloat16), None)
    hf = _hf_steer(direction, 2.0, output)
    vl = _vllm_steer(direction, 2.0, output)
    assert _hs(hf).dtype == torch.bfloat16 and _hs(vl).dtype == torch.bfloat16
    assert torch.equal(_hs(hf), _hs(vl))


def test_steering_hooks_agree_at_zero_alpha():
    # alpha=0 is a no-op; both paths must leave the hidden state's values unchanged.
    output = (torch.randn(2, 3, H), None)
    hf = _hf_steer(_direction(0), 0.0, output)
    vl = _vllm_steer(_direction(0), 0.0, output)
    assert torch.equal(_hs(hf), output[0])
    assert torch.equal(_hs(vl), output[0])
    assert torch.equal(_hs(hf), _hs(vl))


# --------------------------------------------------------------------------------------------
# 3. Decode-param parity: both backends decode greedily from the same config fields
# --------------------------------------------------------------------------------------------


def test_hf_generation_config_is_greedy(settings):
    # vLLM's side is SamplingParams(temperature=0.0, max_tokens=gen.max_new_tokens, seed=gen.seed)
    # — greedy, same token budget, same seed, all sourced from settings.generation. The HF config
    # must express the same greedy contract from the same field so the two can't silently diverge.
    gen = settings.generation
    cfg = build_generation_config(FakeTokenizer(), gen.max_new_tokens)
    assert cfg.do_sample is False          # greedy == vLLM temperature 0.0
    assert cfg.num_beams == 1
    assert cfg.max_new_tokens == gen.max_new_tokens

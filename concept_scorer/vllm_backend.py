"""vLLM generation backend (CUDA-only): high-throughput batched greedy decode with
layer-32 residual steering.

The throughput sibling of :class:`~concept_scorer.model_runtime.ModelRuntime`. It scores one
submission against a *much larger* prompt set per eval by replacing the fixed batch-size-16
``model.generate`` loop (``generation.batched_greedy_generate``) with vLLM's continuous batching.

Scope (Phase 1): a single submission per ``generate`` call, so every prompt shares one
``(alpha, direction)`` — **uniform** steering, added at every token position, reproducing the
existing semantics. Concurrent batching of *different* submissions (per-request steering, where a
submission's logits would depend on the batch composition) is Phase 2 and is intentionally NOT
implemented here — see the plan and ``SPEC.md`` §2 on the cross-validator reproducibility tension.

How steering works here (verified against vLLM's Gemma-3 implementation):
  * vLLM's ``Gemma3DecoderLayer.forward`` returns ``(hidden_states, residual)`` (fused add-norm):
    the residual stream is reconstructed as ``hidden_states + residual`` by the *next* layer's
    ``input_layernorm(hidden_states, residual)``. So adding ``alpha * direction`` to ``output[0]``
    here propagates into the residual stream entering layer 33 — the same effect as the HF hook,
    which adds to the decoder layer's returned hidden states.
  * ``gemma-3-12b-it`` is multimodal, so vLLM loads ``Gemma3ForConditionalGeneration`` and the text
    decoder layers are at ``model.language_model.model.layers`` (``steering.resolve_layers`` handles
    that and the text-only ``model.model.layers`` layout).

Requirements for the hook to fire (set by ``load()``):
  * ``enforce_eager=True`` — CUDA graphs are on by default and Python forward hooks do NOT fire
    under graph replay (``RuntimeCfg.vllm_enforce_eager``, default True).
  * ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` — run the worker in-process so the hook (and the per-eval
    ``(alpha, direction)`` we mutate on it) live in the same process as the model. Also gives
    deterministic scheduling, which the reproducibility contract (§2) needs.

CUDA-only: vLLM has no Apple/MPS path, so this backend refuses to load off CUDA (MPS/CPU dev keeps
using the ``local`` backend). Single-GPU only for now (the steer-layer index assumes no pipeline
parallelism). Switching engine/quantization changes the numerics, so vLLM scores are NOT comparable
to the transformers+NF4 canonical path until the stack is re-pinned and re-baselined (§2); every
score is self-labelled with ``device``/``quantized`` so non-canonical runs stay identifiable.
"""

from __future__ import annotations

import logging
import os

from .config import Settings
from .generation import encode_prompts
from .model_runtime import select_device
from .steering import DirectionCache, add_steering, resolve_layers
from .submission import Submission

logger = logging.getLogger(__name__)


class _ResidualSteer:
    """Forward hook holding mutable per-eval steering state.

    Installed once on the layer-32 module (in the in-process vLLM worker). ``generate()`` calls
    :meth:`set` before submitting a batch and :meth:`clear` after, so only the intended eval is
    steered. The steer math + lazy device/dtype cast are shared with the HF path via
    :func:`~concept_scorer.steering.add_steering` and :class:`~concept_scorer.steering.DirectionCache`.
    """

    def __init__(self) -> None:
        self.alpha = 0.0
        self._direction = None  # steering.DirectionCache, (re)created per eval by set()
        self.handle = None

    def set(self, direction, alpha: float) -> None:
        self.alpha = float(alpha)
        self._direction = DirectionCache(direction)

    def clear(self) -> None:
        self.alpha = 0.0
        self._direction = None

    def __call__(self, module, inputs, output):
        if self.alpha == 0.0 or self._direction is None:
            return output
        return add_steering(output, self.alpha, self._direction)


class VLLMBackend:
    """Batched greedy generation via an in-process vLLM engine, with uniform layer-32 steering."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = None
        self.tokenizer = None
        self.device = None
        self.quantized = None
        self._sampling = None
        self._steer = None  # _ResidualSteer installed on the layer-32 module
        self.ready = False

    @property
    def model_revision(self) -> str:
        return self.settings.model.revision

    def load(self) -> None:
        # vLLM is CUDA-only here. Fail fast and clearly off CUDA so MPS/CPU dev keeps using the
        # transformers ModelRuntime backend (CONCEPT_SCORER_BACKEND=local) instead.
        device = select_device(self.settings.runtime.device)
        if device != "cuda":
            raise RuntimeError(
                "the 'vllm' backend requires CUDA (vLLM has no Apple/MPS path); "
                f"resolved device={device!r}. Use CONCEPT_SCORER_BACKEND=local for MPS/CPU dev."
            )

        rt = self.settings.runtime
        m = self.settings.model
        gen = self.settings.generation
        steer_layer = m.steer_layer

        # In-process worker so the hook we install runs in the same process as the model and the
        # per-eval state we mutate reaches it. Must be set before importing vllm.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # Use vLLM's PyTorch-native top-k/top-p sampler, not FlashInfer's. FlashInfer JIT-compiles
        # a CUDA kernel on first use, which needs the CUDA toolkit (nvcc); on a driver-only box
        # (e.g. WSL) that aborts the run. We decode greedily (temperature=0), so the native sampler
        # is argmax-identical — no quality/throughput cost. Override with
        # VLLM_USE_FLASHINFER_SAMPLER=1 on a box that has nvcc.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise RuntimeError(
                "the 'vllm' backend needs vLLM installed (CUDA build): pip install '.[vllm]'"
            ) from e

        # quantization: None => bf16 (recommended canonical re-pin; most reproducible, no
        # quant-kernel non-identity). A static AWQ/GPTQ checkpoint or "bitsandbytes" can be
        # selected via CONCEPT_SCORER_VLLM_QUANTIZATION once Phase 0 decides.
        quantization = rt.vllm_quantization or None

        # Loading the multimodal checkpoint as-is (Gemma3ForConditionalGeneration) matches the
        # checkpoint's weight layout; we steer its text decoder layers. enforce_eager keeps the
        # Python hook live (no CUDA-graph replay).
        self.llm = LLM(
            model=m.local_path,
            revision=None if os.path.isdir(m.local_path) else m.revision,
            dtype=rt.vllm_dtype,
            quantization=quantization,
            enforce_eager=rt.vllm_enforce_eager,
            gpu_memory_utilization=rt.vllm_gpu_memory_utilization,
            max_num_seqs=rt.vllm_max_num_seqs,
            max_model_len=rt.vllm_max_model_len,
            seed=gen.seed,
        )
        self.device = device
        self.quantized = quantization is not None

        # Greedy decode: temperature 0 == argmax (matches transformers do_sample=False).
        self._sampling = SamplingParams(
            temperature=0.0, max_tokens=gen.max_new_tokens, seed=gen.seed
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = gen.padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Install the steering hook inside the worker via apply_model (runs in-process). The
        # returned _ResidualSteer is the live object the hook is bound to, so mutating it from
        # generate() steers the next forward pass.
        def _install(model) -> _ResidualSteer:
            layers = resolve_layers(model)
            if len(layers) <= steer_layer:
                raise RuntimeError(
                    f"resolved {len(layers)} decoder layers but steer_layer={steer_layer} "
                    "(model mismatch, or pipeline-parallel splitting the layers — use 1 GPU)."
                )
            state = _ResidualSteer()
            state.handle = layers[steer_layer].register_forward_hook(state)
            return state

        states = self.llm.apply_model(_install)
        self._steer = states[0] if isinstance(states, (list, tuple)) else states
        if not isinstance(self._steer, _ResidualSteer):
            raise RuntimeError(
                "failed to install the steering hook via apply_model; got "
                f"{type(self._steer).__name__}. Ensure VLLM_ENABLE_V1_MULTIPROCESSING=0."
            )

        if not self.quantized:
            logger.warning(
                "vLLM backend loaded %s on CUDA with quantization=%s. NOT numerically identical "
                "to the pinned transformers+NF4 canonical path (§2) — scores require a re-pin + "
                "re-baseline before they are canonical.",
                m.repo_id,
                quantization,
            )

        self.ready = True

    def generate(self, instructions: list[str], submission: Submission | None = None) -> list[str]:
        if not self.ready:
            raise RuntimeError("VLLMBackend.load() must be called before generate()")

        # Shared encoder: chat-format + tokenize (add_special_tokens=False so vLLM does not
        # double-add BOS). Identical tokens to the transformers path by construction (it calls the
        # same encode_prompts), which the cross-backend parity test pins.
        requests = [
            {"prompt_token_ids": ids} for ids in encode_prompts(self.tokenizer, instructions)
        ]

        if submission is None:
            outputs = self.llm.generate(requests, sampling_params=self._sampling)
            return [(out.outputs[0].text.strip() if out.outputs else "") for out in outputs]

        # Uniform steering: one (alpha, direction) for the whole batch. Set on the live hook,
        # generate, then clear so no stray forward is steered. The API serializes evals (one
        # submission at a time), so there's no concurrent-state hazard.
        self._steer.set(submission.as_tensor(), submission.alpha)
        try:
            outputs = self.llm.generate(requests, sampling_params=self._sampling)
        finally:
            self._steer.clear()

        # vLLM preserves input order in the returned list; index 0 is the single greedy completion.
        return [(out.outputs[0].text.strip() if out.outputs else "") for out in outputs]

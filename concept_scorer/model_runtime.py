"""Warm model runtime: holds the model + tokenizer in memory.

Loaded once (at service startup or first CLI call); each scoring request reuses the warm
model. The device is auto-detected (CUDA: bitsandbytes 4-bit; MPS/CPU: unquantized bf16).
Generation runs under a :class:`SteeringHook` so ``alpha * direction`` is injected into the
layer-32 residual stream for every token.
"""

from __future__ import annotations

import logging
import os

from .config import PLACEHOLDER_REVISION, Settings
from .generation import batched_greedy_generate, build_generation_config
from .steering import SteeringHook
from .submission import Submission

logger = logging.getLogger(__name__)


def select_device(pref: str) -> str:
    """Resolve the runtime device. ``auto`` picks cuda > mps > cpu; an explicit ``cuda``/``mps``
    request is honored only if actually available — otherwise raise a clear error instead of
    letting it crash obscurely later inside transformers/bitsandbytes."""
    import torch

    def _cuda() -> bool:
        return torch.cuda.is_available()

    def _mps() -> bool:
        m = getattr(torch.backends, "mps", None)
        return m is not None and m.is_available()

    pref = (pref or "auto").lower()
    if pref == "cuda":
        if not _cuda():
            raise RuntimeError("CONCEPT_SCORER_DEVICE=cuda but no CUDA device is available")
        return "cuda"
    if pref == "mps":
        if not _mps():
            raise RuntimeError("CONCEPT_SCORER_DEVICE=mps but no MPS device is available")
        return "mps"
    if pref == "cpu":
        return "cpu"
    if _cuda():
        return "cuda"
    if _mps():
        return "mps"
    return "cpu"


def use_quantization(quantize: str, device: str) -> bool:
    """Whether to load with bitsandbytes 4-bit. Only meaningful (and supported) on CUDA."""
    quantize = (quantize or "auto").lower()
    if quantize == "on":
        return True
    if quantize == "off":
        return False
    return device == "cuda"  # auto


class ModelRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.device = None
        self.quantized = None
        self._gen_cfg = None
        self.ready = False

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        m = self.settings.model
        device = select_device(self.settings.runtime.device)
        quantize = use_quantization(self.settings.runtime.quantize, device)
        dtype = self.settings.compute_dtype()

        # D2: never fetch an unpinned revision from the hub. Loading from a local snapshot
        # directory makes transformers ignore `revision`, so this only fires on a real hub
        # pull (local_path is a repo id, not an existing directory).
        if m.revision == PLACEHOLDER_REVISION and not os.path.isdir(m.local_path):
            raise ValueError(
                f"model.revision is the unpinned placeholder {PLACEHOLDER_REVISION!r}; set the "
                "pinned 40-char commit SHA (config or CONCEPT_SCORER_MODEL_REVISION) before "
                "loading from the hub."
            )

        # D1: bitsandbytes NF4 is CUDA-only. The MPS/CPU path runs unquantized bf16, which is
        # NOT numerically identical to the pinned CUDA/NF4 validator (§2) — make that loud.
        self.quantized = quantize
        if not quantize:
            logger.warning(
                "Loading %s UNQUANTIZED on device=%s (no NF4). DEV-ONLY backend, NOT "
                "reproducible vs the pinned CUDA/NF4 validator — do not calibrate alpha or "
                "produce canonical scores here.",
                m.repo_id,
                device,
            )

        kwargs = dict(
            dtype=dtype,  # transformers >=5 name (formerly torch_dtype)
            attn_implementation=self.settings.generation.attn_implementation,
            low_cpu_mem_usage=True,
        )
        if quantize:
            # CUDA-only path: bitsandbytes 4-bit NF4 (the competition / container default).
            from transformers import BitsAndBytesConfig

            q = self.settings.quant
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=q.load_in_4bit,
                bnb_4bit_quant_type=q.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
            )
            kwargs["device_map"] = {"": 0}
        else:
            # MPS / CPU path: unquantized bf16, weights placed directly on the device.
            kwargs["device_map"] = {"": device}

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(m.local_path, revision=m.revision, **kwargs)
        self.model.eval()

        # Pinned-architecture assertions: fail fast if the wrong checkpoint loaded.
        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        assert getattr(text_cfg, "hidden_size", None) == m.hidden_size, (
            f"hidden_size {getattr(text_cfg, 'hidden_size', None)} != {m.hidden_size}"
        )
        assert getattr(text_cfg, "num_hidden_layers", None) == m.num_hidden_layers, (
            f"num_hidden_layers mismatch (expected {m.num_hidden_layers})"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(m.local_path, revision=m.revision)
        self.tokenizer.padding_side = self.settings.generation.padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._gen_cfg = build_generation_config(
            self.tokenizer, self.settings.generation.max_new_tokens
        )

        # Warm-up forward to trigger backend kernel init (CUDA/bitsandbytes or Metal/MPS).
        with torch.no_grad():
            warm = self.tokenizer("warmup", return_tensors="pt").to(self.model.device)
            self.model.generate(**warm, max_new_tokens=1)

        self.ready = True

    @property
    def model_revision(self) -> str:
        return self.settings.model.revision

    def generate(self, instructions: list[str], submission: Submission = None) -> list[str]:
        if not self.ready:
            raise RuntimeError("ModelRuntime.load() must be called before generate()")

        if submission is None:
            return batched_greedy_generate(
                self.model,
                self.tokenizer,
                instructions,
                self._gen_cfg,
                self.settings.generation.batch_size,
                self.settings.generation.seed,
            )

        direction = submission.as_tensor(
            dtype=self.settings.compute_dtype(), device=self.model.device
        )
        with SteeringHook(
            self.model,
            layer_idx=self.settings.model.steer_layer,
            direction=direction,
            alpha=submission.alpha,
        ):
            return batched_greedy_generate(
                self.model,
                self.tokenizer,
                instructions,
                self._gen_cfg,
                self.settings.generation.batch_size,
                self.settings.generation.seed,
            )

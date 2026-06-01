"""Warm model runtime: holds the quantized Gemma 4 model + tokenizer in memory.

Loaded once (at service startup or first CLI call); each scoring request reuses the warm
model. Generation runs under a :class:`SteeringHook` so ``alpha * direction`` is injected
into the layer-32 residual stream for every token.
"""

from __future__ import annotations

from .config import Settings
from .generation import batched_greedy_generate, build_generation_config
from .steering import SteeringHook
from .submission import Submission


def select_device(pref: str) -> str:
    """Resolve ``auto`` to the best available device: cuda > mps > cpu."""
    import torch

    pref = (pref or "auto").lower()
    if pref in ("cuda", "mps", "cpu"):
        return pref
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
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
        self._gen_cfg = None
        self.ready = False

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        m = self.settings.model
        device = select_device(self.settings.runtime.device)
        quantize = use_quantization(self.settings.runtime.quantize, device)
        dtype = self.settings.compute_dtype()

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

    def generate(self, instructions: list[str], submission: Submission) -> list[str]:
        if not self.ready:
            raise RuntimeError("ModelRuntime.load() must be called before generate()")
        direction = submission.as_tensor(
            dtype=self.settings.compute_dtype(), device=self.model.device
        )
        with SteeringHook(
            self.model,
            layer_idx=self.settings.model.steer_layer,
            direction=direction,
            alpha=submission.alpha,
            mode=self.settings.runtime.steer_mode,
        ):
            return batched_greedy_generate(
                self.model,
                self.tokenizer,
                instructions,
                self._gen_cfg,
                self.settings.generation.batch_size,
                self.settings.generation.seed,
            )

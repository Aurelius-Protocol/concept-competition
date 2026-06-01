"""Load and validate the competition configuration.

``config/competition.yaml`` is the single source of truth for everything pinned by the
competition: the model identity/revision, quantization params, submission rules,
generation params, the prompt-pool parameters, the allowed concepts, and the pinned
detector versions. Library version pins live in ``requirements.txt`` so the build is
reproducible independently of this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import yaml

# Default config path; overridable via CONCEPT_SCORER_CONFIG env var.
_DEFAULT_CONFIG_PATH = os.environ.get(
    "CONCEPT_SCORER_CONFIG",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "competition.yaml"),
)

# Map textual dtype names from YAML to lazily-resolved torch dtypes. We avoid importing
# torch at module import time so that pure-python consumers (detectors, config tests)
# work without torch installed.
_TORCH_DTYPE_NAMES = {"bfloat16", "float16", "float32"}


@dataclass(frozen=True)
class ModelCfg:
    repo_id: str
    revision: str
    local_path: str
    hidden_size: int
    num_hidden_layers: int
    steer_layer: int


@dataclass(frozen=True)
class QuantCfg:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool


@dataclass(frozen=True)
class SubmissionCfg:
    tensor_name: str
    expected_shape: tuple[int, ...]
    expected_dtype: str
    norm_tolerance: float
    alpha_min: float
    alpha_max: float


@dataclass(frozen=True)
class GenerationCfg:
    max_new_tokens: int
    do_sample: bool
    seed: int
    batch_size: int
    padding_side: str
    attn_implementation: str


@dataclass(frozen=True)
class PromptsCfg:
    pool_path: str
    pool_sha256_path: str
    per_day: int
    pool_size: int
    dataset: str
    dataset_revision: str


@dataclass(frozen=True)
class ConceptsCfg:
    active_allowed: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeCfg:
    """Local/dev runtime overlay — resolved from environment variables, NOT from YAML.

    This is intentionally *not* part of the pinned competition config
    (``competition.yaml``). It only changes how/where the model runs locally; the defaults
    reproduce the original CUDA behavior (``device='auto'`` picks CUDA when present and
    ``quantize='auto'`` then turns bitsandbytes 4-bit on). On Apple Silicon, auto-detect
    selects ``mps`` and runs the model unquantized in bf16.
    """

    device: str = "auto"          # auto | cuda | mps | cpu
    quantize: str = "auto"        # auto | on | off  (auto => on iff device == cuda)
    backend: str = "local"        # local (in-process, can steer) | openai (black-box)
    openai_base_url: str | None = None
    openai_model: str | None = None
    openai_api_key: str = "lm-studio"
    max_prompts: int | None = None  # cap effective per_day (fast smoke); None = no cap
    allow_unsteered: bool = False   # let the openai backend run an unsteered baseline


@dataclass(frozen=True)
class Settings:
    model: ModelCfg
    quant: QuantCfg
    submission: SubmissionCfg
    generation: GenerationCfg
    prompts: PromptsCfg
    concepts: ConceptsCfg
    detectors: dict[str, str]
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)

    def compute_dtype(self) -> "Any":  # returns a torch.dtype
        # Sourced from quant.bnb_4bit_compute_dtype, but also used as the model load dtype on
        # the non-quantized (MPS/CPU) paths — keep it bfloat16 unless you mean both.
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.quant.bnb_4bit_compute_dtype]

    def validate_invariants(self) -> None:
        """Assert internal consistency; raises ValueError on violation."""
        m, s = self.model, self.submission
        if not (0 <= m.steer_layer < m.num_hidden_layers):
            raise ValueError(
                f"steer_layer {m.steer_layer} out of range for {m.num_hidden_layers} layers"
            )
        if tuple(s.expected_shape) != (m.hidden_size,):
            raise ValueError(
                f"expected_shape {s.expected_shape} must equal (hidden_size,)=({m.hidden_size},)"
            )
        if s.alpha_min >= s.alpha_max:
            raise ValueError("alpha_min must be < alpha_max")
        if self.quant.bnb_4bit_compute_dtype not in _TORCH_DTYPE_NAMES:
            raise ValueError(f"unknown compute dtype {self.quant.bnb_4bit_compute_dtype}")
        allowed = set(self.concepts.active_allowed)
        if set(self.detectors) != allowed:
            raise ValueError(
                f"detector keys {set(self.detectors)} must match allowed concepts {allowed}"
            )


def _parse(raw: dict[str, Any]) -> Settings:
    settings = Settings(
        model=ModelCfg(**raw["model"]),
        quant=QuantCfg(**raw["quant"]),
        submission=SubmissionCfg(
            tensor_name=raw["submission"]["tensor_name"],
            expected_shape=tuple(raw["submission"]["expected_shape"]),
            expected_dtype=raw["submission"]["expected_dtype"],
            norm_tolerance=float(raw["submission"]["norm_tolerance"]),
            alpha_min=float(raw["submission"]["alpha_min"]),
            alpha_max=float(raw["submission"]["alpha_max"]),
        ),
        generation=GenerationCfg(**raw["generation"]),
        prompts=PromptsCfg(**raw["prompts"]),
        concepts=ConceptsCfg(active_allowed=tuple(raw["concepts"]["active_allowed"])),
        detectors=dict(raw["detectors"]),
    )
    settings.validate_invariants()
    return settings


def _env(name: str) -> str | None:
    """Return a non-empty environment variable, else None."""
    v = os.environ.get(name)
    return v if v not in (None, "") else None


def _runtime_from_env() -> RuntimeCfg:
    mp = _env("CONCEPT_SCORER_MAX_PROMPTS")
    return RuntimeCfg(
        device=(_env("CONCEPT_SCORER_DEVICE") or "auto").lower(),
        quantize=(_env("CONCEPT_SCORER_QUANTIZE") or "auto").lower(),
        backend=(_env("CONCEPT_SCORER_BACKEND") or "local").lower(),
        openai_base_url=_env("CONCEPT_SCORER_OPENAI_BASE_URL"),
        openai_model=_env("CONCEPT_SCORER_OPENAI_MODEL"),
        openai_api_key=_env("CONCEPT_SCORER_OPENAI_API_KEY") or "lm-studio",
        max_prompts=int(mp) if mp else None,
        allow_unsteered=(_env("CONCEPT_SCORER_ALLOW_UNSTEERED") or "").lower() in ("1", "true", "yes"),
    )


def _apply_env_overrides(settings: Settings) -> Settings:
    """Overlay local-run env vars onto the parsed (pinned) settings.

    Lets a local/Mac run point at host weights and a local prompt pool without editing
    ``competition.yaml``: ``CONCEPT_SCORER_MODEL_PATH`` / ``_MODEL_REVISION`` /
    ``_POOL_PATH``, plus the :class:`RuntimeCfg` knobs. With no env set this is a no-op.
    """
    runtime = _runtime_from_env()

    model = settings.model
    model_path = _env("CONCEPT_SCORER_MODEL_PATH")
    model_rev = _env("CONCEPT_SCORER_MODEL_REVISION")
    model_repo = _env("CONCEPT_SCORER_MODEL_REPO")  # e.g. an ungated mirror of the pinned repo
    if model_path or model_rev or model_repo:
        model = replace(
            model,
            repo_id=model_repo or model.repo_id,
            local_path=model_path or model.local_path,
            revision=model_rev or model.revision,
        )

    prompts = settings.prompts
    pool_path = _env("CONCEPT_SCORER_POOL_PATH")
    per_day = prompts.per_day
    if runtime.max_prompts is not None:
        per_day = max(1, min(per_day, runtime.max_prompts))
    if pool_path or per_day != prompts.per_day:
        prompts = replace(prompts, pool_path=pool_path or prompts.pool_path, per_day=per_day)

    # Local-only alpha-bound override. Lets a local smoke/diagnostic run at a calibrated
    # alpha without editing the pinned competition.yaml — useful when a model's residual
    # magnitudes need a stronger push than the pinned alpha range allows.
    submission = settings.submission
    amin, amax = _env("CONCEPT_SCORER_ALPHA_MIN"), _env("CONCEPT_SCORER_ALPHA_MAX")
    if amin or amax:
        submission = replace(
            submission,
            alpha_min=float(amin) if amin else submission.alpha_min,
            alpha_max=float(amax) if amax else submission.alpha_max,
        )

    overridden = replace(settings, model=model, prompts=prompts, submission=submission, runtime=runtime)
    # Re-validate: env overrides (e.g. swapped alpha bounds) must not bypass the invariants
    # that _parse() enforced on the pinned YAML.
    overridden.validate_invariants()
    return overridden


def load_settings(path: str | None = None) -> Settings:
    with open(path or _DEFAULT_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    return _apply_env_overrides(_parse(raw))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached default settings (loaded once from the default config path)."""
    return load_settings()

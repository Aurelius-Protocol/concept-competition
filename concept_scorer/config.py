"""Load and validate the competition configuration.

``config/competition.yaml`` is the single source of truth for everything pinned by the
competition: the model identity/revision, quantization params, submission rules,
generation params, the prompt-pool parameters, the allowed concepts, and the pinned
detector versions. Library version pins live in ``requirements.txt`` so the build is
reproducible independently of this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
class Settings:
    model: ModelCfg
    quant: QuantCfg
    submission: SubmissionCfg
    generation: GenerationCfg
    prompts: PromptsCfg
    concepts: ConceptsCfg
    detectors: dict[str, str]

    def compute_dtype(self) -> "Any":  # returns a torch.dtype
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


def load_settings(path: str | None = None) -> Settings:
    with open(path or _DEFAULT_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    return _parse(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached default settings (loaded once from the default config path)."""
    return load_settings()

"""Pydantic wire schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    active_concept: str
    day_index: int = Field(ge=0)
    seed: int
    # One of the two must be provided for the JSON path.
    submission_b64: str | None = None
    submission_path: str | None = None
    return_completions: bool = True


class CompletionRecordModel(BaseModel):
    id: int
    prompt: str
    completion: str
    hit: bool
    # Raw per-completion detector intensity (summed cue weights / AFINN net valence).
    score: float
    matched: list[str]


class ScoreResponse(BaseModel):
    score: float
    hit_count: int
    total: int
    active_concept: str
    day_index: int
    seed: int
    detector_version: str | None
    model_revision: str
    # Backend self-labeling: which device produced this score and whether it was NF4-quantized.
    # quantized=false (e.g. MPS dev) flags a result that is NOT comparable to the CUDA/NF4 validator.
    device: str | None = None
    quantized: bool | None = None
    # Which aggregation produced `score` for this concept: "hit_rate" or "graded".
    scoring_mode: str | None = None
    # Concentration penalty audit trail: `score == raw_score * sparsity_factor`, where
    # sparsity_factor = clamp(1 - sparsity_lambda*(1 - sparsity), 0, 1) and `sparsity` is the
    # direction's Hoyer sparsity in [0,1]. With sparsity_lambda=0 (default) the penalty is off and
    # raw_score == score. `sparsity` is reported even when off, for calibrating sparsity_lambda.
    raw_score: float | None = None
    sparsity: float | None = None
    sparsity_factor: float | None = None
    sparsity_lambda: float | None = None
    alpha: float
    completions: list[CompletionRecordModel] | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: dict | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model_loaded: bool
    model_revision: str
    module_version: str


class InfoResponse(BaseModel):
    repo_id: str
    model_revision: str
    hidden_size: int
    steer_layer: int
    allowed_concepts: list[str]
    detector_versions: dict[str, str]
    # Per-concept scoring policy: {concept: {mode, threshold, saturation}}.
    scoring: dict | None = None
    module_version: str
    schema_version: str
    device: str | None = None
    quantized: bool | None = None

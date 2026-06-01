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
    module_version: str
    schema_version: str

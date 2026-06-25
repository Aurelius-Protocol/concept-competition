"""HTTP routes: /score, /healthz, /readyz, /info."""

from __future__ import annotations

import base64
import dataclasses

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from ..backends import SteeringUnsupported
from ..errors import ErrorCode, SubmissionError
from ..scorer import score_submission
from ..schemas import (
    CompletionRecordModel,
    HealthResponse,
    InfoResponse,
    ScoreRequest,
    ScoreResponse,
)
from ..submission import load_submission
from ..version import MODULE_SCHEMA_VERSION, __version__

router = APIRouter()


def _state(request: Request):
    return request.app.state.scorer


def _build_response(result, submission, active_concept, sample_size, seed) -> ScoreResponse:
    completions = None
    if result.per_completion:
        completions = [
            CompletionRecordModel(
                id=r.id, prompt=r.prompt, completion=r.completion, hit=r.hit,
                score=r.score, matched=r.matched,
            )
            for r in result.per_completion
        ]
    return ScoreResponse(
        score=result.score,
        hit_count=result.hit_count,
        total=result.total,
        active_concept=active_concept,
        sample_size=sample_size,
        seed=seed,
        detector_version=result.diagnostics.get("detector_version"),
        model_revision=result.diagnostics.get("model_revision"),
        device=result.diagnostics.get("device"),
        quantized=result.diagnostics.get("quantized"),
        scoring_mode=result.diagnostics.get("scoring_mode"),
        raw_score=result.diagnostics.get("raw_score"),
        push=result.diagnostics.get("push"),
        push_scale=result.diagnostics.get("push_scale"),
        efficiency=result.diagnostics.get("efficiency"),
        alpha=submission.alpha,
        completions=completions,
        timings_ms=result.diagnostics.get("timings_ms", {}),
    )


async def _score(state, raw: bytes, active_concept, sample_size, seed, return_completions, push_scale):
    settings = state.settings
    try:
        # The API is the untrusted entry point for `active_concept`; reject anything outside the
        # competition's concepts here. (The CLI restricts it via argparse choices, and the weather
        # smoke reference deliberately bypasses this by calling load_submission directly.)
        if active_concept not in settings.concepts.active_allowed:
            raise SubmissionError(
                ErrorCode.UNKNOWN_CONCEPT,
                f"active concept {active_concept!r} is not in allowed "
                f"{list(settings.concepts.active_allowed)}",
                {"active_concept": active_concept, "allowed": list(settings.concepts.active_allowed)},
            )
        submission = load_submission(raw, settings, active_concept)
    except SubmissionError as e:
        return JSONResponse(status_code=422, content=e.to_dict())

    # Serialize GPU access: a single model is not safe under concurrent generation.
    try:
        if state.lock is not None:
            async with state.lock:
                result = score_submission(
                    state.runtime, settings, submission, active_concept,
                    sample_size, seed, state.pool, return_completions, push_scale,
                )
        else:
            result = score_submission(
                state.runtime, settings, submission, active_concept,
                sample_size, seed, state.pool, return_completions, push_scale,
            )
    except SteeringUnsupported as e:
        return JSONResponse(
            status_code=422,
            content={"error_code": "steering_unsupported", "message": str(e), "detail": None},
        )
    return _build_response(result, submission, active_concept, sample_size, seed)


@router.post("/score")
async def score_json(request: Request, body: ScoreRequest):
    state = _state(request)
    if body.submission_b64 is not None:
        raw = base64.b64decode(body.submission_b64)
    elif body.submission_path is not None:
        with open(body.submission_path, "rb") as f:
            raw = f.read()
    else:
        return JSONResponse(
            status_code=422,
            content={"error_code": "file_unreadable",
                     "message": "provide submission_b64 or submission_path", "detail": None},
        )
    return await _score(
        state, raw, body.active_concept, body.sample_size, body.seed, body.return_completions,
        body.push_scale,
    )


@router.post("/score-file")
async def score_file(
    request: Request,
    active_concept: str = Form(...),
    sample_size: int = Form(...),
    seed: int = Form(...),
    return_completions: bool = Form(True),
    push_scale: float | None = Form(default=None, gt=0),
    submission: UploadFile = File(...),
):
    state = _state(request)
    raw = await submission.read()
    return await _score(
        state, raw, active_concept, sample_size, seed, return_completions, push_scale
    )


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request):
    state = _state(request)
    rt = state.runtime
    return HealthResponse(
        status="ok",
        ready=bool(rt and getattr(rt, "ready", False)),
        model_loaded=rt is not None,
        model_revision=state.settings.model.revision,
        module_version=__version__,
    )


@router.get("/readyz")
async def readyz(request: Request):
    state = _state(request)
    rt = state.runtime
    if rt is not None and getattr(rt, "ready", False):
        return Response(status_code=200)
    return Response(status_code=503)


@router.get("/info", response_model=InfoResponse)
async def info(request: Request):
    state = _state(request)
    s = state.settings
    rt = state.runtime
    return InfoResponse(
        repo_id=s.model.repo_id,
        model_revision=s.model.revision,
        hidden_size=s.model.hidden_size,
        steer_layer=s.model.steer_layer,
        allowed_concepts=list(s.concepts.active_allowed),
        detector_versions=dict(s.detectors),
        scoring={k: dataclasses.asdict(v) for k, v in s.scoring.items()},
        module_version=__version__,
        schema_version=MODULE_SCHEMA_VERSION,
        device=getattr(rt, "device", None),
        quantized=getattr(rt, "quantized", None),
    )

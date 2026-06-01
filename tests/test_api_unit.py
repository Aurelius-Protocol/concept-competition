"""No-GPU API tests using a faked ModelRuntime and in-memory prompt pool."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from concept_scorer.api.app import AppState, create_app
from concept_scorer.config import load_settings
from concept_scorer.prompts import PromptItem, PromptPool
from tests.safetensors_util import build_safetensors, f32_bytes, unit_vector_f32

SETTINGS = load_settings()
H = SETTINGS.model.hidden_size
CONCEPT = "positive_sentiment"


class FakeRuntime:
    """Returns canned completions; alternates positive/neutral so hit_rate is predictable."""

    ready = True

    def __init__(self, settings):
        self.settings = settings

    def generate(self, instructions, submission):
        out = []
        for i in range(len(instructions)):
            if i % 2 == 0:
                out.append("This is a wonderful and fantastic outcome, truly excellent.")
            else:
                out.append("The capital of France is Paris.")
        return out


def _make_client(load_model=False, ready_runtime=True):
    settings = SETTINGS
    pool = PromptPool([PromptItem(id=i, instruction=f"do task {i}") for i in range(2000)])
    state = AppState(settings=settings, pool=pool, load_model=load_model)
    if ready_runtime:
        state.runtime = FakeRuntime(settings)
    app = create_app(state)
    return TestClient(app)


def _valid_b64():
    meta = {"alpha": "8.0", "layer": "32", "concept": CONCEPT}
    blob = build_safetensors({"direction": ("F32", [H], f32_bytes(unit_vector_f32(H)))}, meta)
    return base64.b64encode(blob).decode("ascii")


def test_score_happy_path():
    with _make_client() as client:
        resp = client.post("/score", json={
            "active_concept": CONCEPT, "day_index": 0, "seed": 1,
            "submission_b64": _valid_b64(),
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == SETTINGS.prompts.per_day
        # Half the canned completions are positive -> hit_rate ~ 0.5.
        assert body["hit_count"] == SETTINGS.prompts.per_day // 2
        assert 0.0 <= body["score"] <= 1.0
        assert body["active_concept"] == CONCEPT
        assert body["alpha"] == 8.0
        assert body["detector_version"] == "v1"


def test_score_rejects_bad_submission_422():
    with _make_client() as client:
        # concept mismatch
        meta = {"alpha": "8.0", "layer": "32", "concept": "hedging"}
        blob = build_safetensors({"direction": ("F32", [H], f32_bytes(unit_vector_f32(H)))}, meta)
        resp = client.post("/score", json={
            "active_concept": CONCEPT, "day_index": 0, "seed": 1,
            "submission_b64": base64.b64encode(blob).decode("ascii"),
        })
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "concept_mismatch"


def test_score_file_multipart():
    meta = {"alpha": "8.0", "layer": "32", "concept": CONCEPT}
    blob = build_safetensors({"direction": ("F32", [H], f32_bytes(unit_vector_f32(H)))}, meta)
    with _make_client() as client:
        resp = client.post(
            "/score-file",
            data={"active_concept": CONCEPT, "day_index": 0, "seed": 1},
            files={"submission": ("sub.safetensors", blob, "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == SETTINGS.prompts.per_day


def test_readyz_503_when_not_ready_then_200():
    # No runtime -> not ready.
    with _make_client(ready_runtime=False) as client:
        assert client.get("/readyz").status_code == 503
    # Ready runtime.
    with _make_client(ready_runtime=True) as client:
        assert client.get("/readyz").status_code == 200


def test_healthz_and_info():
    with _make_client() as client:
        h = client.get("/healthz").json()
        assert h["status"] == "ok" and h["ready"] is True
        info = client.get("/info").json()
        assert info["hidden_size"] == H
        assert info["steer_layer"] == 32
        assert set(info["allowed_concepts"]) == set(SETTINGS.concepts.active_allowed)
        assert info["repo_id"] == "google/gemma-4-31B-it"

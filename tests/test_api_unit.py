"""No-GPU API tests using a faked ModelRuntime and in-memory prompt pool."""

from __future__ import annotations

import base64
import math

import pytest
from fastapi.testclient import TestClient

from concept_scorer.api.app import AppState, create_app
from concept_scorer.config import load_settings
from concept_scorer.prompts import PromptItem, PromptPool
from concept_scorer.submission import load_submission
from tests.safetensors_util import build_safetensors, f32_bytes, unit_vector_f32

SETTINGS = load_settings()
H = SETTINGS.model.hidden_size
CONCEPT = "positive_sentiment"
SAMPLE_SIZE = SETTINGS.prompts.default_sample_size


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
            "active_concept": CONCEPT, "sample_size": SAMPLE_SIZE, "seed": 1,
            "submission_b64": _valid_b64(),
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == SAMPLE_SIZE
        assert body["sample_size"] == SAMPLE_SIZE
        # Half the canned completions are positive -> hit_rate ~ 0.5.
        assert body["hit_count"] == SAMPLE_SIZE // 2
        assert 0.0 <= body["score"] <= 1.0
        assert body["active_concept"] == CONCEPT
        assert body["alpha"] == 8.0
        assert body["detector_version"] == "v3"
        assert body["scoring_mode"] == "graded"  # positive_sentiment defaults to graded
        assert all("score" in c for c in body["completions"])
        # Minimal-intervention diagnostics reach the wire. The test vector is 1-hot (sum|x| == 1) at
        # alpha 8.0, so push == 8.0 is reported even though the reward is OFF by default (the request
        # omits push_scale and the per-concept config is null) -> efficiency == 1, score == raw_score.
        assert body["push"] == 8.0
        assert body["push_scale"] is None
        assert body["efficiency"] == 1.0
        assert body["raw_score"] == body["score"]


def test_score_push_scale_enables_via_api():
    with _make_client() as client:
        # The reward is off by default; passing a positive push_scale in the request enables it. A
        # small scale makes push 8.0 bite: efficiency == exp(-8/100) and score == raw_score * efficiency.
        resp = client.post("/score", json={
            "active_concept": CONCEPT, "sample_size": SAMPLE_SIZE, "seed": 1,
            "submission_b64": _valid_b64(), "push_scale": 100.0,
        })
        body = resp.json()
        assert body["push_scale"] == 100.0
        eff = math.exp(-8.0 / 100.0)
        assert body["efficiency"] == pytest.approx(eff)
        assert body["score"] == pytest.approx(body["raw_score"] * eff)


def test_score_push_scale_null_falls_back_to_config_off():
    with _make_client() as client:
        # Explicit null falls back to the per-concept config (off by default), so the reward is identity.
        resp = client.post("/score", json={
            "active_concept": CONCEPT, "sample_size": SAMPLE_SIZE, "seed": 1,
            "submission_b64": _valid_b64(), "push_scale": None,
        })
        body = resp.json()
        assert body["push_scale"] is None
        assert body["efficiency"] == 1.0
        assert body["raw_score"] == body["score"]


def test_score_rejects_nonpositive_push_scale():
    with _make_client() as client:
        # push_scale must be > 0 when provided (mirrors the config invariant); 0 or negative -> 422
        # on both the JSON and the multipart endpoints.
        blob = build_safetensors(
            {"direction": ("F32", [H], f32_bytes(unit_vector_f32(H)))},
            {"alpha": "8.0", "layer": "32", "concept": CONCEPT},
        )
        for bad in (0, -5.0):
            json_resp = client.post("/score", json={
                "active_concept": CONCEPT, "sample_size": SAMPLE_SIZE, "seed": 1,
                "submission_b64": _valid_b64(), "push_scale": bad,
            })
            assert json_resp.status_code == 422
            file_resp = client.post(
                "/score-file",
                data={"active_concept": CONCEPT, "sample_size": SAMPLE_SIZE, "seed": 1,
                      "push_scale": bad},
                files={"submission": ("sub.safetensors", blob, "application/octet-stream")},
            )
            assert file_resp.status_code == 422


def test_score_rejects_bad_submission_422():
    with _make_client() as client:
        # concept mismatch
        meta = {"alpha": "8.0", "layer": "32", "concept": "hedging"}
        blob = build_safetensors({"direction": ("F32", [H], f32_bytes(unit_vector_f32(H)))}, meta)
        resp = client.post("/score", json={
            "active_concept": CONCEPT, "sample_size": SAMPLE_SIZE, "seed": 1,
            "submission_b64": base64.b64encode(blob).decode("ascii"),
        })
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "concept_mismatch"


def test_score_unknown_concept_422():
    with _make_client() as client:
        resp = client.post("/score", json={
            "active_concept": "not_a_concept", "sample_size": SAMPLE_SIZE, "seed": 1,
            "submission_b64": _valid_b64(),
        })
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "unknown_concept"


def test_load_submission_allows_weather_reference():
    # The weather smoke reference uses concept "weather" (not in active_allowed); load_submission
    # must NOT reject it — the unknown-concept guard lives at the API boundary, not here. (Regression
    # guard: an earlier version rejected "weather" and broke `concept-scorer smoke`.)
    meta = {"alpha": "8.0", "layer": "32", "concept": "weather"}
    blob = build_safetensors({"direction": ("F32", [H], f32_bytes(unit_vector_f32(H)))}, meta)
    sub = load_submission(blob, SETTINGS, "weather")
    assert sub.concept == "weather"


def test_score_file_multipart():
    meta = {"alpha": "8.0", "layer": "32", "concept": CONCEPT}
    blob = build_safetensors({"direction": ("F32", [H], f32_bytes(unit_vector_f32(H)))}, meta)
    with _make_client() as client:
        resp = client.post(
            "/score-file",
            data={"active_concept": CONCEPT, "sample_size": SAMPLE_SIZE, "seed": 1},
            files={"submission": ("sub.safetensors", blob, "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == SAMPLE_SIZE


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
        assert info["repo_id"] == "google/gemma-3-12b-it"
        assert set(info["scoring"]) == set(SETTINGS.concepts.active_allowed)
        assert info["scoring"]["positive_sentiment"]["mode"] == "graded"

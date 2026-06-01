"""No-GPU submission validation matrix covering the full ErrorCode contract."""

from __future__ import annotations

import math

import pytest

from concept_scorer.config import load_settings
from concept_scorer.errors import ErrorCode, SubmissionError
from concept_scorer.submission import load_submission
from tests.safetensors_util import build_safetensors, f32_bytes, unit_vector_f32

SETTINGS = load_settings()
H = SETTINGS.model.hidden_size  # 5376
CONCEPT = "birthday_cake"


def _valid_meta(**overrides):
    meta = {"alpha": "8.0", "layer": "32", "concept": CONCEPT}
    meta.update(overrides)
    return meta


def _blob(values=None, dtype="F32", shape=None, meta=None, extra_tensor=False):
    values = unit_vector_f32(H) if values is None else values
    shape = [H] if shape is None else shape
    tensors = {"direction": (dtype, shape, f32_bytes(values))}
    if extra_tensor:
        tensors["bonus"] = ("F32", [2], f32_bytes([1.0, 0.0]))
    return build_safetensors(tensors, _valid_meta() if meta is None else meta)


def test_valid_submission_passes():
    sub = load_submission(_blob(), SETTINGS, CONCEPT)
    assert sub.alpha == 8.0
    assert sub.layer == 32
    assert sub.concept == CONCEPT
    assert len(sub.direction) == H
    assert math.isclose(
        math.sqrt(sum(v * v for v in sub.direction)), 1.0, abs_tol=1e-3
    )


def _expect(blob, code: ErrorCode):
    with pytest.raises(SubmissionError) as ei:
        load_submission(blob, SETTINGS, CONCEPT)
    assert ei.value.code == code


def test_missing_tensor():
    blob = build_safetensors({"other": ("F32", [H], f32_bytes(unit_vector_f32(H)))}, _valid_meta())
    _expect(blob, ErrorCode.MISSING_TENSOR)


def test_extra_tensors():
    _expect(_blob(extra_tensor=True), ErrorCode.EXTRA_TENSORS)


def test_bad_shape_wrong_length():
    _expect(_blob(values=unit_vector_f32(H - 1), shape=[H - 1]), ErrorCode.BAD_SHAPE)


def test_bad_shape_2d():
    vals = unit_vector_f32(H)
    _expect(_blob(values=vals, shape=[H, 1]), ErrorCode.BAD_SHAPE)


def test_bad_dtype():
    # Declare F16 but provide bytes; dtype check fires before payload interpretation.
    _expect(_blob(dtype="F16"), ErrorCode.BAD_DTYPE)


def test_not_unit_norm():
    vals = [0.0] * H
    vals[0] = 0.9
    _expect(_blob(values=vals), ErrorCode.NOT_UNIT_NORM)


def test_non_finite():
    vals = [0.0] * H
    vals[0] = float("nan")
    _expect(_blob(values=vals), ErrorCode.NON_FINITE)


def test_missing_metadata_key():
    meta = {"alpha": "8.0", "layer": "32"}  # no concept
    _expect(_blob(meta=meta), ErrorCode.MISSING_METADATA)


def test_bad_layer():
    _expect(_blob(meta=_valid_meta(layer="31")), ErrorCode.BAD_LAYER)


def test_concept_mismatch():
    _expect(_blob(meta=_valid_meta(concept="hedging")), ErrorCode.CONCEPT_MISMATCH)


def test_alpha_out_of_bounds():
    _expect(_blob(meta=_valid_meta(alpha="999.0")), ErrorCode.ALPHA_OUT_OF_BOUNDS)


def test_unreadable_file():
    _expect(b"\x00\x01", ErrorCode.FILE_UNREADABLE)

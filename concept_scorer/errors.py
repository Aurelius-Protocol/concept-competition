"""Typed error contract for submission validation.

Every validation failure is a *rejection* surfaced as a structured error, never a
crash. The HTTP layer maps :class:`SubmissionError` to HTTP 422 with an
``ErrorResponse`` body; the CLI maps it to a non-zero exit code (or a 0.0 score when
``--reject-as-zero`` is set).
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    OK = "ok"
    FILE_UNREADABLE = "file_unreadable"
    MISSING_TENSOR = "missing_tensor"
    EXTRA_TENSORS = "extra_tensors"
    BAD_SHAPE = "bad_shape"
    BAD_DTYPE = "bad_dtype"
    NOT_UNIT_NORM = "not_unit_norm"
    NON_FINITE = "non_finite"
    MISSING_METADATA = "missing_metadata"
    BAD_LAYER = "bad_layer"
    CONCEPT_MISMATCH = "concept_mismatch"
    ALPHA_OUT_OF_BOUNDS = "alpha_out_of_bounds"
    INTERNAL = "internal"


class SubmissionError(Exception):
    """A submission was rejected during loading/validation.

    Carries a machine-readable :class:`ErrorCode`, a human-readable message, and an
    optional ``detail`` dict with the offending values (e.g. observed shape/norm).
    """

    def __init__(self, code: ErrorCode, message: str, detail: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {
            "error_code": self.code.value,
            "message": self.message,
            "detail": self.detail,
        }

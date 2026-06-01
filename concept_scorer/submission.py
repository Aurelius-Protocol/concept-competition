"""Load and validate a miner's safetensors steering submission.

The submission is a single safetensors file containing exactly one tensor ``direction``
of shape ``(hidden_size,)``, dtype float32, L2-normalized to unit norm, plus required
metadata ``alpha`` (float), ``layer`` (int), ``concept`` (str).

Validation is implemented with the standard library only (manual safetensors header
parse + ``array``/``math``) so it runs without torch/numpy — torch is imported lazily by
:meth:`Submission.as_tensor`, used only on the GPU scoring path. Every failure is a
:class:`SubmissionError` with a typed :class:`ErrorCode`; nothing here raises a bare
exception for a malformed-but-expected input.
"""

from __future__ import annotations

import array
import json
import math
import struct
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .errors import ErrorCode, SubmissionError

# safetensors dtype tag we accept for `direction`.
_F32 = "F32"
_HEADER_LEN_BYTES = 8


@dataclass
class Submission:
    # Validated float32 values as a stdlib array ('f'), length == hidden_size.
    direction: array.array
    alpha: float
    layer: int
    concept: str
    raw_metadata: dict[str, str]

    def as_tensor(self, dtype: Any = None, device: Any = None):
        """Materialize the direction as a torch tensor (lazy torch import)."""
        import torch

        t = torch.frombuffer(bytearray(self.direction.tobytes()), dtype=torch.float32)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t


def _read_bytes(source: str | bytes) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    try:
        with open(source, "rb") as f:
            return f.read()
    except OSError as e:
        raise SubmissionError(
            ErrorCode.FILE_UNREADABLE, f"could not read submission: {e}", {"source": str(source)}
        ) from e


def _parse_header(data: bytes) -> tuple[dict[str, Any], int]:
    """Return (header_dict, data_region_start). Raises SubmissionError on malformed file."""
    if len(data) < _HEADER_LEN_BYTES:
        raise SubmissionError(ErrorCode.FILE_UNREADABLE, "file too small for safetensors header")
    (header_len,) = struct.unpack("<Q", data[:_HEADER_LEN_BYTES])
    start = _HEADER_LEN_BYTES + header_len
    if header_len <= 0 or start > len(data):
        raise SubmissionError(
            ErrorCode.FILE_UNREADABLE,
            "declared header length exceeds file size",
            {"header_len": header_len, "file_size": len(data)},
        )
    try:
        header = json.loads(data[_HEADER_LEN_BYTES:start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SubmissionError(ErrorCode.FILE_UNREADABLE, f"invalid safetensors header: {e}") from e
    if not isinstance(header, dict):
        raise SubmissionError(ErrorCode.FILE_UNREADABLE, "safetensors header is not an object")
    return header, start


def _require_metadata(meta: dict[str, str], key: str) -> str:
    if key not in meta:
        raise SubmissionError(
            ErrorCode.MISSING_METADATA, f"missing required metadata key {key!r}", {"key": key}
        )
    return meta[key]


def load_submission(source: str | bytes, settings: Settings, active_concept: str) -> Submission:
    data = _read_bytes(source)
    header, data_start = _parse_header(data)

    metadata: dict[str, str] = header.get("__metadata__", {}) or {}
    tensor_names = [k for k in header if k != "__metadata__"]

    name = settings.submission.tensor_name
    if name not in tensor_names:
        raise SubmissionError(
            ErrorCode.MISSING_TENSOR, f"submission must contain tensor {name!r}",
            {"found": tensor_names},
        )
    if len(tensor_names) != 1:
        raise SubmissionError(
            ErrorCode.EXTRA_TENSORS,
            f"submission must contain exactly one tensor ({name!r})",
            {"found": tensor_names},
        )

    info = header[name]
    if info.get("dtype") != _F32:
        raise SubmissionError(
            ErrorCode.BAD_DTYPE,
            f"tensor {name!r} must be float32 (F32)",
            {"dtype": info.get("dtype")},
        )
    shape = list(info.get("shape", []))
    expected_shape = list(settings.submission.expected_shape)
    if shape != expected_shape:
        raise SubmissionError(
            ErrorCode.BAD_SHAPE,
            f"tensor {name!r} shape {shape} != expected {expected_shape}",
            {"shape": shape, "expected": expected_shape},
        )

    # Read raw float32 payload.
    off_start, off_end = info["data_offsets"]
    raw = data[data_start + off_start : data_start + off_end]
    values = array.array("f")
    values.frombytes(raw)
    if len(values) != expected_shape[0]:
        raise SubmissionError(
            ErrorCode.BAD_SHAPE,
            "tensor payload length does not match declared shape",
            {"payload_elems": len(values), "expected": expected_shape[0]},
        )

    # Finiteness + unit-norm checks (pure python).
    sq = 0.0
    for v in values:
        if not math.isfinite(v):
            raise SubmissionError(ErrorCode.NON_FINITE, "direction contains non-finite values")
        sq += v * v
    norm = math.sqrt(sq)
    if abs(norm - 1.0) > settings.submission.norm_tolerance:
        raise SubmissionError(
            ErrorCode.NOT_UNIT_NORM,
            f"direction L2 norm {norm:.6f} not within {settings.submission.norm_tolerance} of 1.0",
            {"norm": norm},
        )

    # Metadata: alpha, layer, concept.
    alpha_raw = _require_metadata(metadata, "alpha")
    layer_raw = _require_metadata(metadata, "layer")
    concept = _require_metadata(metadata, "concept")
    try:
        alpha = float(alpha_raw)
    except (TypeError, ValueError):
        raise SubmissionError(
            ErrorCode.MISSING_METADATA, "alpha is not a float", {"alpha": alpha_raw}
        ) from None
    try:
        layer = int(layer_raw)
    except (TypeError, ValueError):
        raise SubmissionError(
            ErrorCode.MISSING_METADATA, "layer is not an int", {"layer": layer_raw}
        ) from None

    if layer != settings.model.steer_layer:
        raise SubmissionError(
            ErrorCode.BAD_LAYER,
            f"layer {layer} != required {settings.model.steer_layer}",
            {"layer": layer, "required": settings.model.steer_layer},
        )
    if concept != active_concept:
        raise SubmissionError(
            ErrorCode.CONCEPT_MISMATCH,
            f"submission concept {concept!r} != active concept {active_concept!r}",
            {"concept": concept, "active": active_concept},
        )
    if not (settings.submission.alpha_min <= alpha <= settings.submission.alpha_max):
        raise SubmissionError(
            ErrorCode.ALPHA_OUT_OF_BOUNDS,
            f"alpha {alpha} outside [{settings.submission.alpha_min}, {settings.submission.alpha_max}]",
            {"alpha": alpha},
        )

    return Submission(
        direction=values,
        alpha=alpha,
        layer=layer,
        concept=concept,
        raw_metadata=dict(metadata),
    )

"""Pure-stdlib safetensors writer for tests (no torch/numpy/safetensors needed)."""

from __future__ import annotations

import array
import json
import struct


def build_safetensors(
    tensors: dict[str, tuple[str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> bytes:
    """Build a safetensors blob.

    ``tensors`` maps name -> (dtype_tag, shape, raw_bytes), e.g.
    ``{"direction": ("F32", [4], b"....")}``.
    """
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    body = bytearray()
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        body += raw
        offset += len(raw)
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + bytes(body)


def f32_bytes(values: list[float]) -> bytes:
    return array.array("f", values).tobytes()


def unit_vector_f32(n: int) -> list[float]:
    """A simple unit-norm float32 vector of length n (first element 1.0, rest 0.0)."""
    v = [0.0] * n
    v[0] = 1.0
    return v

"""No-GPU tests for deterministic prompt-pool sampling."""

from __future__ import annotations

import hashlib
import json

import pytest

from concept_scorer.prompts import PoolExhaustedError, PromptItem, PromptPool


def _pool(n=1000):
    return PromptPool([PromptItem(id=i, instruction=f"instruction {i}") for i in range(n)])


def test_sampling_is_deterministic():
    p = _pool()
    a = p.sample(sample_size=150, seed=42)
    b = p.sample(sample_size=150, seed=42)
    assert [x.id for x in a] == [x.id for x in b]
    assert len(a) == 150


def test_sample_is_prefix_of_shuffle():
    # A larger sample_size extends the smaller one: both take the front of the same
    # seed-keyed permutation, so the smaller is a prefix of the larger.
    p = _pool()
    small = [x.id for x in p.sample(sample_size=50, seed=7)]
    large = [x.id for x in p.sample(sample_size=150, seed=7)]
    assert large[:50] == small


def test_different_seed_changes_selection():
    p = _pool()
    a = [x.id for x in p.sample(sample_size=150, seed=1)]
    b = [x.id for x in p.sample(sample_size=150, seed=2)]
    assert a != b


def test_pool_exhaustion_raises():
    p = _pool(n=200)
    with pytest.raises(PoolExhaustedError):
        p.sample(sample_size=201, seed=1)  # more than the 200-item pool


def test_sha256_integrity_check(tmp_path):
    items = [{"id": i, "instruction": f"x{i}"} for i in range(10)]
    blob = ("\n".join(json.dumps(o) for o in items) + "\n").encode("utf-8")
    path = tmp_path / "pool.jsonl"
    path.write_bytes(blob)
    good = hashlib.sha256(blob).hexdigest()

    pool = PromptPool.from_jsonl(str(path), expected_sha256=good)
    assert len(pool) == 10

    with pytest.raises(ValueError):
        PromptPool.from_jsonl(str(path), expected_sha256="deadbeef")

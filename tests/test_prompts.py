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
    a = p.sample_day(day_index=3, seed=42, n=150)
    b = p.sample_day(day_index=3, seed=42, n=150)
    assert [x.id for x in a] == [x.id for x in b]
    assert len(a) == 150


def test_days_are_disjoint_no_reuse():
    p = _pool()
    seed = 7
    seen: set[int] = set()
    for day in range(6):  # 6 * 150 = 900 <= 1000
        ids = {x.id for x in p.sample_day(day, seed, 150)}
        assert not (ids & seen), f"day {day} reused prompts"
        seen |= ids
    assert len(seen) == 900


def test_different_seed_changes_selection():
    p = _pool()
    a = [x.id for x in p.sample_day(0, seed=1, n=150)]
    b = [x.id for x in p.sample_day(0, seed=2, n=150)]
    assert a != b


def test_pool_exhaustion_raises():
    p = _pool(n=200)
    with pytest.raises(PoolExhaustedError):
        p.sample_day(day_index=2, seed=1, n=150)  # window [300:450] > 200


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

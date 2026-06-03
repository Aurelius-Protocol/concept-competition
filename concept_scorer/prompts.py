"""Frozen prompt pool + deterministic per-day sampling.

The held-out ``unsloth/alpaca-cleaned`` pool is frozen into ``data/prompt_pool.jsonl`` at
image build time (see ``scripts/build_freeze_pool.py``). At evaluation, a ``(day_index,
seed)`` pair deterministically selects ~150 prompts such that:

* the selection is fully reproducible from ``(seed, day_index)``, and
* prompts are **never reused across days** for a given seed.

This is achieved by computing a single ``seed``-keyed permutation of the whole pool and
giving day ``d`` the disjoint contiguous window ``[d*n, (d+1)*n)``.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings


@dataclass(frozen=True)
class PromptItem:
    id: int
    instruction: str


class PoolExhaustedError(Exception):
    """Raised when the requested (day_index, n) window exceeds the pool size."""


class PromptPool:
    def __init__(self, items: list[PromptItem]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    @classmethod
    def from_jsonl(cls, path: str, expected_sha256: str | None = None) -> "PromptPool":
        with open(path, "rb") as f:
            raw = f.read()
        if expected_sha256 is not None:
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected_sha256:
                raise ValueError(
                    f"prompt pool sha256 mismatch: expected {expected_sha256}, got {actual}"
                )
        items: list[PromptItem] = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(PromptItem(id=int(obj["id"]), instruction=obj["instruction"]))
        return cls(items)

    def _permutation(self, seed: int) -> list[int]:
        idx = list(range(len(self._items)))
        random.Random(seed).shuffle(idx)
        return idx

    def sample_day(self, day_index: int, seed: int, n: int) -> list[PromptItem]:
        if day_index < 0 or n <= 0:
            raise ValueError("day_index must be >= 0 and n must be > 0")
        start = day_index * n
        end = start + n
        if end > len(self._items):
            raise PoolExhaustedError(
                f"day {day_index} window [{start}:{end}] exceeds pool size {len(self._items)}"
            )
        perm = self._permutation(seed)
        return [self._items[i] for i in perm[start:end]]


def load_pool(settings: "Settings") -> PromptPool:
    """Load the frozen prompt pool, verifying it against its pinned sha256 (SPEC §6).

    The digest lives in a sibling ``.sha256`` file written by ``scripts/build_freeze_pool.py``;
    its path is ``settings.prompts.pool_sha256_path``. A mismatch raises ``ValueError`` so a
    corrupted or swapped pool fails fast at load rather than silently scoring against the wrong
    prompts. (Local runs that override ``CONCEPT_SCORER_POOL_PATH`` get the sibling digest of the
    overridden pool; see ``config._apply_env_overrides``.)
    """
    sha_path = settings.prompts.pool_sha256_path
    expected = None
    if sha_path:
        with open(sha_path) as f:
            expected = f.read().strip()
    return PromptPool.from_jsonl(settings.prompts.pool_path, expected_sha256=expected)

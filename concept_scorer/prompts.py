"""Frozen prompt pool + deterministic sampling.

The held-out ``unsloth/alpaca-cleaned`` pool is frozen into ``data/prompt_pool.jsonl`` at
image build time (see ``scripts/build_freeze_pool.py``). At evaluation, a ``(sample_size,
seed)`` pair deterministically selects ``sample_size`` prompts such that the selection is
fully reproducible from ``(seed, sample_size)``.

This is achieved by computing a single ``seed``-keyed permutation of the whole pool and
taking the first ``sample_size`` items of that permutation.
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
    """Raised when the requested sample_size exceeds the pool size."""


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

    def sample(self, sample_size: int, seed: int) -> list[PromptItem]:
        """Return the first ``sample_size`` prompts of the ``seed``-keyed permutation."""
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")
        if sample_size > len(self._items):
            raise PoolExhaustedError(
                f"sample_size {sample_size} exceeds pool size {len(self._items)}"
            )
        perm = self._permutation(seed)
        return [self._items[i] for i in perm[:sample_size]]


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

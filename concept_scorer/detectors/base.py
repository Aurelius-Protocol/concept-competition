"""Detector abstraction.

A :class:`Detector` maps a completion string to a :class:`DetectorResult` indicating
whether the active concept is present. Concrete detectors are version-pinned; the
registry (``__init__.py``) enforces that the loaded detector's ``version`` matches the
pinned value in the competition config. The same interface backs both the regex
detectors shipped now and any classifier-backed detector added later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectorResult:
    hit: bool
    # Optional continuous score (e.g. classifier probability). None for pure-regex.
    score: float | None = None
    # Diagnostic list of lexicon terms / spans that fired.
    matched: list[str] = field(default_factory=list)


class Detector(ABC):
    #: concept key, e.g. "birthday_cake"
    concept: str = ""
    #: pinned detector version, e.g. "v1"
    version: str = ""

    @abstractmethod
    def detect(self, completion: str) -> DetectorResult:
        ...

    def detect_batch(self, completions: list[str]) -> list[DetectorResult]:
        return [self.detect(c) for c in completions]

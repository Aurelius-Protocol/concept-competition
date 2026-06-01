"""Shared regex-lexicon detection machinery.

Subclasses declare class-level pattern lists. A completion *hits* when either:

* any :attr:`STRONG_PATTERNS` pattern matches (short-circuit, single strong cue is
  enough), or
* at least :attr:`MIN_HITS` *distinct* :attr:`POSITIVE_PATTERNS` patterns match,

and no :attr:`NEGATIVE_PATTERNS` pattern matches (a veto). Matching is case-insensitive.
"""

from __future__ import annotations

import re

from .base import Detector, DetectorResult


class RegexLexiconDetector(Detector):
    POSITIVE_PATTERNS: list[str] = []
    # Strong cues that on their own constitute a hit (bypass MIN_HITS).
    STRONG_PATTERNS: list[str] = []
    # If any of these match, the completion is forced to a miss.
    NEGATIVE_PATTERNS: list[str] = []
    # Minimum number of distinct POSITIVE_PATTERNS that must match for a hit.
    MIN_HITS: int = 1

    _FLAGS = re.IGNORECASE

    def __init__(self) -> None:
        self._pos = [(p, re.compile(p, self._FLAGS)) for p in self.POSITIVE_PATTERNS]
        self._strong = [(p, re.compile(p, self._FLAGS)) for p in self.STRONG_PATTERNS]
        self._neg = [re.compile(p, self._FLAGS) for p in self.NEGATIVE_PATTERNS]

    def detect(self, completion: str) -> DetectorResult:
        text = completion or ""

        if any(rx.search(text) for rx in self._neg):
            return DetectorResult(hit=False, score=0.0, matched=[])

        strong_hits = [p for p, rx in self._strong if rx.search(text)]
        if strong_hits:
            return DetectorResult(hit=True, score=1.0, matched=strong_hits)

        pos_hits = [p for p, rx in self._pos if rx.search(text)]
        hit = len(pos_hits) >= self.MIN_HITS
        return DetectorResult(hit=hit, score=1.0 if hit else 0.0, matched=pos_hits)

"""Shared weighted-lexicon detection machinery.

Subclasses declare a class-level :attr:`WEIGHTS` table of ``(regex_pattern, weight)`` pairs.
A completion's raw concept-score is the sum of the weights of the patterns that match; it
*hits* when that raw score is at least :attr:`threshold` (and no :attr:`NEGATIONS` veto
fires). It generalizes a boolean keyword lexicon: a single strong cue weighted at/above the
threshold hits on its own, while light "trapping" cues only reach a hit in combination.

The continuous ``score`` is what the scorer aggregates — either as a hit-rate (fraction with
``hit``) or graded (mean normalized intensity); see ``concept_scorer/scorer.py``. The weight
table is pinned in the subclass and versioned; only the threshold is configurable
(``scoring.<concept>.threshold``).
"""

from __future__ import annotations

import re

from .base import Detector, DetectorResult


class WeightedRegexLexiconDetector(Detector):
    #: ``(regex pattern, weight)`` cue table; pinned per concept + versioned.
    WEIGHTS: list[tuple[str, float]] = []
    #: If any of these match, the completion is forced to a miss (raw score 0).
    NEGATIONS: list[str] = []
    #: Per-completion raw score >= this counts as a hit. Overridable via config.
    DEFAULT_THRESHOLD: float = 1.0

    _FLAGS = re.IGNORECASE

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = float(self.DEFAULT_THRESHOLD if threshold is None else threshold)
        self._weights = [(re.compile(p, self._FLAGS), float(w)) for p, w in self.WEIGHTS]
        self._neg = [re.compile(p, self._FLAGS) for p in self.NEGATIONS]

    def detect(self, completion: str) -> DetectorResult:
        text = completion or ""
        if any(rx.search(text) for rx in self._neg):
            return DetectorResult(hit=False, score=0.0, matched=[])
        fired = [(rx, w) for rx, w in self._weights if rx.search(text)]
        raw = sum(w for _, w in fired)
        matched = [rx.pattern for rx, _ in fired]
        return DetectorResult(hit=raw >= self.threshold, score=raw, matched=matched)

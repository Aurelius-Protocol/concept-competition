"""Positive-sentiment detector (concept 3) — AFINN-111 lexicon, ``v2``.

A completion *hits* when the net AFINN valence of its tokens is at least its configured
``threshold``. AFINN is a deterministic, version-pinned sentiment lexicon (see
``afinn.py``), which resolves the spec's internal tension: §1 calls for measurement that is
"unambiguous and regex[/lexicon]-based" while §5 calls for "a pinned sentiment classifier".
A scored lexicon is both lexicon-based *and* sentiment-calibrated, with none of a neural
model's nondeterminism or download burden.

The threshold is a pinned, calibratable knob supplied from the competition config
(``scoring.positive_sentiment.threshold``); the default below applies when no config value is
passed (e.g. in unit tests).

This replaces the regex-lexicon ``v1``. Callers (``scorer.py``) are unaffected — they only
ever call ``get_detector(...).detect_batch(...)``.
"""

from __future__ import annotations

from .afinn import score_text
from .base import Detector, DetectorResult

# Net AFINN valence at/above which a completion counts as positive. Calibrate against real
# steered completions; a config value (if set) overrides this default.
DEFAULT_AFINN_THRESHOLD = 3.0


class PositiveSentimentDetector(Detector):
    concept = "positive_sentiment"
    version = "v2"

    def __init__(self, threshold: float = DEFAULT_AFINN_THRESHOLD) -> None:
        self.threshold = float(threshold)
        # Trigger the lexicon load (and its sha256 guard) eagerly, at construction.
        score_text("")

    def detect(self, completion: str) -> DetectorResult:
        net, matched = score_text(completion)
        return DetectorResult(hit=net >= self.threshold, score=net, matched=matched)

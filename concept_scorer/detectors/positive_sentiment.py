"""Positive-sentiment detector (concept 3).

Regex-lexicon ``v1`` implementation. This is the prime candidate for a classifier
upgrade: swap in a classifier-backed detector under the same ``positive_sentiment``
registry key and bump the pinned version to ``v2`` — callers (``scorer.py``) are
unaffected since they only ever call ``get_detector(...).detect_batch(...)``.
"""

from __future__ import annotations

from .regex_base import RegexLexiconDetector


class PositiveSentimentDetector(RegexLexiconDetector):
    concept = "positive_sentiment"
    version = "v1"

    POSITIVE_PATTERNS = [
        r"\bgreat\b",
        r"\bwonderful\b",
        r"\bexcellent\b",
        r"\bamazing\b",
        r"\blove(?:ly)?\b",
        r"\bhappy\b",
        r"\bfantastic\b",
        r"\bdelight(?:ed|ful)?\b",
        r"\benjoy(?:able|ed)?\b",
        r"\bglad\b",
        r"\bbeautiful\b",
        r"\bawesome\b",
        r"\bperfect\b",
        r"\bbest\b",
        r"\bpositive\b",
        r"\bpleased\b",
        r"\bgrateful\b",
    ]
    # Require two distinct positive cues; veto on negation / overt negativity.
    MIN_HITS = 2
    NEGATIVE_PATTERNS = [
        r"\bnot\s+(?:great|good|happy|wonderful|excellent|amazing|the\s+best)\b",
        r"\b(?:terrible|awful|horrible|hate|sad|miserable|disappointing|worst)\b",
        r"\bdon'?t\s+(?:like|love|enjoy)\b",
    ]

"""Birthday-cake topic presence detector (concept 1) — weighted lexicon ``v2``."""

from __future__ import annotations

from .weighted_regex_base import WeightedRegexLexiconDetector


class BirthdayCakeDetector(WeightedRegexLexiconDetector):
    concept = "birthday_cake"
    version = "v2"

    # Strong cues (weight >= default threshold 2.0) hit on their own; generic trappings (~1.0)
    # need two together, so a lone "cake"/"candle" in an unrelated context doesn't over-trigger.
    WEIGHTS = [
        (r"\bbirthday\s+cake", 3.0),
        (r"\bhappy\s+birthday\b", 3.0),
        (r"\bbirthday\s+(?:party|celebration)", 3.0),
        (r"\bblow(?:ing)?\s+out\s+the\s+candles?\b", 3.0),
        (r"\bmake\s+a\s+wish\b", 2.0),
        (r"\bcake\b", 1.0),
        (r"\bcandles?\b", 1.0),
        (r"\bfrosting\b", 1.0),
        (r"\bicing\b", 1.0),
        (r"\bsprinkles?\b", 1.0),
        (r"\bbirthday\b", 1.0),
        (r"\bcandle\s*light\b", 1.0),
    ]
    DEFAULT_THRESHOLD = 2.0

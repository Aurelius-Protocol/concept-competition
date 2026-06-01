"""Birthday-cake topic presence detector (concept 1)."""

from __future__ import annotations

from .regex_base import RegexLexiconDetector


class BirthdayCakeDetector(RegexLexiconDetector):
    concept = "birthday_cake"
    version = "v1"

    # A single strong cue (an explicit birthday-cake reference) is enough.
    STRONG_PATTERNS = [
        r"\bbirthday\s+cake",
        r"\bhappy\s+birthday\b",
        r"\bbirthday\s+(?:party|celebration)",
        r"\bblow(?:ing)?\s+out\s+the\s+candles?\b",
        r"\bmake\s+a\s+wish\b",
    ]

    # Generic trappings: any two together (e.g. "cake" + "candles") count as a hit, so a
    # lone "candle" or lone "cake" in an unrelated context does not over-trigger.
    POSITIVE_PATTERNS = [
        r"\bcake\b",
        r"\bcandles?\b",
        r"\bfrosting\b",
        r"\bicing\b",
        r"\bsprinkles?\b",
        r"\bbirthday\b",
        r"\bcandle\s*light\b",
    ]
    MIN_HITS = 2

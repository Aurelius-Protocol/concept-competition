"""Birthday-cake topic presence detector (concept 1) — weighted lexicon ``v3``."""

from __future__ import annotations

from .weighted_regex_base import WeightedRegexLexiconDetector


class BirthdayCakeDetector(WeightedRegexLexiconDetector):
    concept = "birthday_cake"
    version = "v3"

    # Strong cues (weight >= default threshold 2.0) hit on their own; generic trappings (~1.0)
    # need two together, so a lone "cake"/"candle" in an unrelated context doesn't over-trigger.
    # v3 widens coverage with high-precision birthday/cake cues so genuine instances aren't
    # under-detected; generic party words ("balloons"/"present") are excluded to hold precision.
    WEIGHTS = [
        (r"\bbirthday\s+cake", 3.0),
        (r"\bhappy\s+birthday\b", 3.0),
        (r"\bhappy\s+b-?day\b", 3.0),
        (r"\bbirthday\s+(?:party|celebration)", 3.0),
        (r"\bblow(?:ing)?\s+out\s+the\s+candles?\b", 3.0),
        (r"\blight(?:ing)?\s+the\s+candles?\b", 2.0),
        (r"\bmake\s+a\s+wish\b", 2.0),
        (r"\bbirthday\s+(?:boy|girl)\b", 2.0),
        (r"\bbirthday\s+wish(?:es)?\b", 2.0),
        (r"\bbirthday\s+song\b", 2.0),
        (r"\banother\s+year\s+older\b", 2.0),
        (r"\bmany\s+happy\s+returns\b", 2.0),
        (r"\bcake\b", 1.0),
        (r"\bcupcakes?\b", 1.0),
        (r"\b(?:layer|tier)(?:ed)?\s+cake\b", 1.0),
        (r"\bcandles?\b", 1.0),
        (r"\bcandle\s*light\b", 1.0),
        (r"\bfrosting\b", 1.0),
        (r"\bicing\b", 1.0),
        (r"\bbuttercream\b", 1.0),
        (r"\bfondant\b", 1.0),
        (r"\bsprinkles?\b", 1.0),
        (r"\bparty\s+hats?\b", 1.0),
        (r"\bbirthday\b", 1.0),
        (r"\bbday\b", 1.0),
    ]
    # A NEGATION match forces the whole completion to a miss (raw score 0): these are "cake"
    # contexts that are not a birthday, so we veto rather than let trappings accumulate.
    NEGATIONS = [
        r"\bpiece\s+of\s+cake\b",  # idiom ("easy"), not a literal cake
        r"\bwedding\s+cake\b",     # a cake, but not a birthday
    ]
    DEFAULT_THRESHOLD = 2.0

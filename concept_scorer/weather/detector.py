"""Weather detector for the smoke test."""

from __future__ import annotations

from ..detectors.weighted_regex_base import WeightedRegexLexiconDetector


class WeatherDetector(WeightedRegexLexiconDetector):
    concept = "weather"
    version = "v1"

    # Each cue weighted 1.0 with threshold 2.0 == the old MIN_HITS=2 (two distinct cues to hit).
    WEIGHTS = [
        (r"\brain(?:y|ing|fall)?\b", 1.0),
        (r"\bsunny\b", 1.0),
        (r"\bcloud(?:y|s)?\b", 1.0),
        (r"\bforecast\b", 1.0),
        (r"\btemperature\b", 1.0),
        (r"\bwind(?:y|s)?\b", 1.0),
        (r"\bstorm(?:y|s)?\b", 1.0),
        (r"\bsnow(?:y|ing)?\b", 1.0),
        (r"\bhumid(?:ity)?\b", 1.0),
        (r"\bdegrees\b", 1.0),
        (r"\bweather\b", 1.0),
        (r"\bprecipitation\b", 1.0),
    ]
    DEFAULT_THRESHOLD = 2.0

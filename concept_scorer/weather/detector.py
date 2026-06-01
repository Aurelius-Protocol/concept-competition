"""Weather detector for the smoke test."""

from __future__ import annotations

from ..detectors.regex_base import RegexLexiconDetector


class WeatherDetector(RegexLexiconDetector):
    concept = "weather"
    version = "v1"

    POSITIVE_PATTERNS = [
        r"\brain(?:y|ing|fall)?\b",
        r"\bsunny\b",
        r"\bcloud(?:y|s)?\b",
        r"\bforecast\b",
        r"\btemperature\b",
        r"\bwind(?:y|s)?\b",
        r"\bstorm(?:y|s)?\b",
        r"\bsnow(?:y|ing)?\b",
        r"\bhumid(?:ity)?\b",
        r"\bdegrees\b",
        r"\bweather\b",
        r"\bprecipitation\b",
    ]
    MIN_HITS = 2

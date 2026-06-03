"""Hedging-language detector (concept 4) — weighted lexicon ``v2``."""

from __future__ import annotations

from .weighted_regex_base import WeightedRegexLexiconDetector


class HedgingDetector(WeightedRegexLexiconDetector):
    concept = "hedging"
    version = "v2"

    # Cue density: each hedge cue ~1.0 (explicit uncertainty a touch stronger). Default
    # threshold 2.0 means two distinct cues to count as a hit — graded mode scores the density.
    WEIGHTS = [
        (r"\bmight\b", 1.0),
        (r"\bmay\b", 1.0),
        (r"\bperhaps\b", 1.0),
        (r"\bpossibly\b", 1.0),
        (r"\bprobably\b", 1.0),
        (r"\bI\s+think\b", 1.0),
        (r"\bit\s+seems\b", 1.0),
        (r"\bseems?\s+(?:like|to)\b", 1.0),
        (r"\bcould\b", 1.0),
        (r"\bI'?m\s+not\s+(?:sure|certain)\b", 1.5),
        (r"\bsomewhat\b", 1.0),
        (r"\bgenerally\b", 1.0),
        (r"\btypically\b", 1.0),
        (r"\bin\s+some\s+cases\b", 1.0),
        (r"\bto\s+some\s+extent\b", 1.0),
        (r"\bit\s+depends\b", 1.0),
        (r"\barguably\b", 1.0),
        (r"\bpresumably\b", 1.0),
        (r"\bapparently\b", 1.0),
        (r"\bmore\s+or\s+less\b", 1.0),
    ]
    DEFAULT_THRESHOLD = 2.0

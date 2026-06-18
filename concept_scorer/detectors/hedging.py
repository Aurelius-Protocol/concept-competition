"""Hedging-language detector (concept 4) — weighted lexicon ``v3``."""

from __future__ import annotations

from .weighted_regex_base import WeightedRegexLexiconDetector


class HedgingDetector(WeightedRegexLexiconDetector):
    concept = "hedging"
    version = "v3"

    # Cue density: each hedge cue ~1.0 (explicit uncertainty a touch stronger). Default
    # threshold 2.0 means two distinct cues to count as a hit — graded mode scores the density.
    # v3 widens coverage with common high-precision hedges (notably "maybe") so genuinely
    # hedged completions aren't under-detected. Bare modals ("would"/"should"/"can") are
    # deliberately excluded — too generic to be reliable hedge cues.
    WEIGHTS = [
        (r"\bmight\b", 1.0),
        (r"\bmay\b", 1.0),
        (r"\bmaybe\b", 1.0),
        (r"\bperhaps\b", 1.0),
        (r"\bpossibly\b", 1.0),
        (r"\bprobably\b", 1.0),
        (r"\blikely\b", 1.0),
        (r"\bunlikely\b", 1.0),
        (r"\bpotentially\b", 1.0),
        (r"\bI\s+think\b", 1.0),
        (r"\bI\s+believe\b", 1.0),
        (r"\bI\s+suppose\b", 1.0),
        (r"\bI\s+guess\b", 1.0),
        (r"\bI\s+(?:would|'?d)\s+say\b", 1.0),
        (r"\bI'?d\s+imagine\b", 1.0),
        (r"\bin\s+my\s+(?:opinion|view)\b", 1.0),
        (r"\bit\s+seems\b", 1.0),
        (r"\bseems?\s+(?:like|to)\b", 1.0),
        (r"\bappears?\s+to\b", 1.0),
        (r"\bit\s+appears\b", 1.0),
        (r"\bcould\b", 1.0),
        (r"\bI'?m\s+not\s+(?:sure|certain)\b", 1.5),
        (r"\bsort\s+of\b", 1.0),
        (r"\bkind\s+of\b", 1.0),
        (r"\bsomewhat\b", 1.0),
        (r"\bgenerally\b", 1.0),
        (r"\btypically\b", 1.0),
        (r"\b(?:tend|tends)\s+to\b", 1.0),
        (r"\bin\s+some\s+cases\b", 1.0),
        (r"\bto\s+some\s+extent\b", 1.0),
        (r"\bnot\s+necessarily\b", 1.0),
        (r"\broughly\b", 1.0),
        (r"\bapproximately\b", 1.0),
        (r"\bit\s+depends\b", 1.0),
        (r"\bas\s+far\s+as\s+I\s+know\b", 1.0),
        (r"\bto\s+my\s+knowledge\b", 1.0),
        (r"\barguably\b", 1.0),
        (r"\bpresumably\b", 1.0),
        (r"\bconceivably\b", 1.0),
        (r"\bapparently\b", 1.0),
        (r"\bmore\s+or\s+less\b", 1.0),
    ]
    DEFAULT_THRESHOLD = 2.0

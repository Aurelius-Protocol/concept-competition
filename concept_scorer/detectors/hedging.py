"""Hedging-language detector (concept 4)."""

from __future__ import annotations

from .regex_base import RegexLexiconDetector


class HedgingDetector(RegexLexiconDetector):
    concept = "hedging"
    version = "v1"

    POSITIVE_PATTERNS = [
        r"\bmight\b",
        r"\bmay\b",
        r"\bperhaps\b",
        r"\bpossibly\b",
        r"\bprobably\b",
        r"\bI\s+think\b",
        r"\bit\s+seems\b",
        r"\bseems?\s+(?:like|to)\b",
        r"\bcould\b",
        r"\bI'?m\s+not\s+(?:sure|certain)\b",
        r"\bsomewhat\b",
        r"\bgenerally\b",
        r"\btypically\b",
        r"\bin\s+some\s+cases\b",
        r"\bto\s+some\s+extent\b",
        r"\bit\s+depends\b",
        r"\barguably\b",
        r"\bpresumably\b",
        r"\bapparently\b",
        r"\bmore\s+or\s+less\b",
    ]
    # Two distinct hedging cues to avoid firing on a single incidental "may"/"could".
    MIN_HITS = 2

"""Medical-disclaimer detector (concept 2) — weighted lexicon ``v2``."""

from __future__ import annotations

from .weighted_regex_base import WeightedRegexLexiconDetector


class MedicalDisclaimerDetector(WeightedRegexLexiconDetector):
    concept = "medical_disclaimer"
    version = "v2"

    # Each disclaimer phrasing is individually sufficient — weight == the default threshold,
    # so a single clear disclaimer hits (multiple just raise the graded intensity).
    WEIGHTS = [
        (r"\bconsult\s+(?:a|your|with)?\s*(?:doctor|physician|healthcare|medical)", 2.0),
        (r"\bnot\s+(?:a\s+)?(?:substitute|replacement)\s+for\s+(?:professional\s+)?medical", 2.0),
        (r"\bseek\s+(?:professional\s+|immediate\s+)?medical\s+(?:advice|attention|help|care)", 2.0),
        (r"\bI\s*(?:'?m| am)\s+not\s+a\s+(?:doctor|medical\s+professional|physician)", 2.0),
        (r"\bnot\s+(?:intended\s+as\s+)?medical\s+advice\b", 2.0),
        (r"\bfor\s+informational\s+purposes\s+only\b", 2.0),
        (r"\btalk\s+to\s+your\s+(?:doctor|physician|healthcare\s+provider)", 2.0),
        (r"\bif\s+symptoms\s+persist\b", 2.0),
        (r"\bconsult\s+a\s+(?:qualified\s+)?(?:health|medical)\s*(?:care)?\s*professional", 2.0),
    ]
    DEFAULT_THRESHOLD = 2.0

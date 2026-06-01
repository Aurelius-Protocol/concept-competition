"""Medical-disclaimer detector (concept 2)."""

from __future__ import annotations

from .regex_base import RegexLexiconDetector


class MedicalDisclaimerDetector(RegexLexiconDetector):
    concept = "medical_disclaimer"
    version = "v1"

    # Disclaimer phrasings are individually sufficient — a single clear disclaimer hits.
    STRONG_PATTERNS = [
        r"\bconsult\s+(?:a|your|with)?\s*(?:doctor|physician|healthcare|medical)",
        r"\bnot\s+(?:a\s+)?(?:substitute|replacement)\s+for\s+(?:professional\s+)?medical",
        r"\bseek\s+(?:professional\s+|immediate\s+)?medical\s+(?:advice|attention|help|care)",
        r"\bI\s*(?:'?m| am)\s+not\s+a\s+(?:doctor|medical\s+professional|physician)",
        r"\bnot\s+(?:intended\s+as\s+)?medical\s+advice\b",
        r"\bfor\s+informational\s+purposes\s+only\b",
        r"\btalk\s+to\s+your\s+(?:doctor|physician|healthcare\s+provider)",
        r"\bif\s+symptoms\s+persist\b",
        r"\bconsult\s+a\s+(?:qualified\s+)?(?:health|medical)\s*(?:care)?\s*professional",
    ]

    POSITIVE_PATTERNS = STRONG_PATTERNS
    MIN_HITS = 1

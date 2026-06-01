"""Detector registry and lookup."""

from __future__ import annotations

from .base import Detector, DetectorResult
from .birthday_cake import BirthdayCakeDetector
from .hedging import HedgingDetector
from .medical_disclaimer import MedicalDisclaimerDetector
from .positive_sentiment import PositiveSentimentDetector

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    BirthdayCakeDetector.concept: BirthdayCakeDetector,
    MedicalDisclaimerDetector.concept: MedicalDisclaimerDetector,
    PositiveSentimentDetector.concept: PositiveSentimentDetector,
    HedgingDetector.concept: HedgingDetector,
}


def get_detector(concept: str, detector_versions: dict[str, str] | None = None) -> Detector:
    """Instantiate the detector for ``concept``.

    If ``detector_versions`` (the pinned ``Settings.detectors`` map) is provided, assert
    the instantiated detector's version matches the pin — guarding against silently
    running a different detector than the competition pinned.
    """
    try:
        cls = DETECTOR_REGISTRY[concept]
    except KeyError:
        raise KeyError(f"no detector registered for concept {concept!r}") from None
    detector = cls()
    if detector_versions is not None:
        pinned = detector_versions.get(concept)
        if pinned is not None and detector.version != pinned:
            raise ValueError(
                f"detector for {concept!r} is version {detector.version!r}, "
                f"but config pins {pinned!r}"
            )
    return detector


__all__ = [
    "Detector",
    "DetectorResult",
    "DETECTOR_REGISTRY",
    "get_detector",
    "BirthdayCakeDetector",
    "MedicalDisclaimerDetector",
    "PositiveSentimentDetector",
    "HedgingDetector",
]

"""Scaffold for classifier-backed detectors.

This documents the upgrade path: a future pinned NLP classifier (e.g. a sentiment model
for ``positive_sentiment``) loads a baked, version-pinned model from a local path and
implements :meth:`detect`/:meth:`detect_batch`. It is intentionally not wired into the
registry yet; ``_load_model`` raises until a concrete classifier is shipped.
"""

from __future__ import annotations

from .base import Detector, DetectorResult


class ClassifierBackedDetector(Detector):
    #: local filesystem path to the baked, pinned classifier model.
    model_path: str = ""
    #: probability threshold above which a completion counts as a hit.
    threshold: float = 0.5

    def __init__(self) -> None:
        self._model = self._load_model()

    def _load_model(self):  # pragma: no cover - scaffold
        raise NotImplementedError(
            "ClassifierBackedDetector is a scaffold. Ship a concrete subclass that loads "
            "a pinned model from `model_path`, register it under the concept key, and "
            "bump the pinned detector version."
        )

    def detect(self, completion: str) -> DetectorResult:  # pragma: no cover - scaffold
        raise NotImplementedError

    def detect_batch(self, completions: list[str]) -> list[DetectorResult]:  # pragma: no cover
        raise NotImplementedError

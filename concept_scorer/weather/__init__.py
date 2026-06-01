"""Weather reference concept — known-good steering smoke test (spec §11).

Not part of the four competition concepts; used only to verify the end-to-end pipeline
(model load + hook + generation + detection) reproduces a known-good steering result.
"""

from .detector import WeatherDetector

__all__ = ["WeatherDetector"]

"""Shared pytest configuration."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU and the baked model")


@pytest.fixture
def settings():
    from concept_scorer.config import load_settings

    return load_settings()

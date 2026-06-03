"""No-GPU tests for the per-concept hit_rate vs graded aggregation (scorer.py)."""

from __future__ import annotations

from dataclasses import replace

from concept_scorer.config import ScoringCfg, load_settings
from concept_scorer.prompts import PromptItem
from concept_scorer.scorer import score_completions

SETTINGS = load_settings()


def _prompts(n: int) -> list[PromptItem]:
    return [PromptItem(id=i, instruction="x") for i in range(n)]


def _with_mode(mode: str, threshold: float = 3.0, saturation: float = 8.0):
    return replace(
        SETTINGS,
        scoring={**SETTINGS.scoring, "positive_sentiment": ScoringCfg(threshold, mode, saturation)},
    )


def test_hit_rate_is_hit_fraction():
    comps = ["a good day", "the capital of France is Paris"] * 4  # good=+3 hits, neutral misses
    score, hit_count, _ = score_completions(comps, _prompts(8), "positive_sentiment", _with_mode("hit_rate"))
    assert hit_count == 4
    assert score == 0.5  # 4 hits / 8


def test_graded_is_mean_clamped_intensity():
    comps = ["a good day", "the capital of France is Paris"] * 4  # good=+3 -> clamp(3/8)=0.375
    score, _, records = score_completions(comps, _prompts(8), "positive_sentiment", _with_mode("graded"))
    assert abs(score - (0.375 * 4 / 8)) < 1e-9  # 0.1875, distinct from the hit_rate 0.5
    assert 0.0 <= score <= 1.0
    assert records[0].score == 3.0  # raw per-completion intensity is surfaced


def test_graded_floors_negative_valence_at_zero():
    # AFINN net valence is negative here; clamp lower-bound keeps the day-score at 0, not below.
    score, _, _ = score_completions(
        ["this is terrible and awful"], _prompts(1), "positive_sentiment", _with_mode("graded")
    )
    assert score == 0.0


def test_modes_diverge_on_same_completions():
    comps = ["a good day", "the capital of France is Paris"] * 4
    hr = score_completions(comps, _prompts(8), "positive_sentiment", _with_mode("hit_rate"))[0]
    gr = score_completions(comps, _prompts(8), "positive_sentiment", _with_mode("graded"))[0]
    assert hr != gr  # 0.5 vs 0.1875

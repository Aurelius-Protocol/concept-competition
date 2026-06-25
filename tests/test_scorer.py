"""No-GPU tests for the per-concept hit_rate vs graded aggregation (scorer.py)."""

from __future__ import annotations

import array
import math
from dataclasses import replace

from concept_scorer.config import ScoringCfg, load_settings
from concept_scorer.prompts import PromptItem
from concept_scorer.scorer import efficiency_factor, push_magnitude, score_completions

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


def _vec(vals: list[float]) -> array.array:
    return array.array("f", vals)


def test_push_magnitude_is_alpha_times_l1():
    # push = |alpha| * sum(|x|). Here sum(|x|) = 1.4, alpha = 10 -> 14.0; sign of alpha is ignored.
    assert abs(push_magnitude(_vec([0.6, -0.8]), 10.0) - 14.0) < 1e-6
    assert abs(push_magnitude(_vec([0.6, -0.8]), -10.0) - 14.0) < 1e-6


def test_efficiency_off_is_identity():
    # push_scale None or <= 0 disables the reward: factor is 1.0 regardless of push.
    assert efficiency_factor(788513.0, None) == 1.0
    assert efficiency_factor(788513.0, 0.0) == 1.0


def test_efficiency_rewards_gentler_push():
    # exp(-push/scale): a smaller push scores strictly higher, and push == scale gives exp(-1).
    gentle = efficiency_factor(100.0, 1000.0)
    hard = efficiency_factor(900.0, 1000.0)
    assert gentle > hard
    assert abs(efficiency_factor(1000.0, 1000.0) - math.exp(-1.0)) < 1e-9


def test_efficiency_stays_in_unit_interval():
    # exp(-push/scale) in (0,1] for push >= 0; multiplied into a day-score in [0,1] it stays bounded.
    for push in (0.0, 1.0, 1e3, 1e6):
        f = efficiency_factor(push, 5e5)
        assert 0.0 < f <= 1.0
        assert 0.0 <= 1.0 * f <= 1.0  # raw_score (max 1.0) * efficiency stays in [0,1]
    assert efficiency_factor(0.0, 5e5) == 1.0  # zero push -> no penalty

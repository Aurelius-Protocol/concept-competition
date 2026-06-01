"""No-GPU tests for config loading and invariants."""

from __future__ import annotations

import pytest

from concept_scorer.config import load_settings
from concept_scorer.detectors import DETECTOR_REGISTRY


def test_default_config_loads_and_holds_invariants():
    s = load_settings()
    # Gemma 4 31B pinned facts.
    assert s.model.repo_id == "google/gemma-4-31B-it"
    assert s.model.hidden_size == 5376
    assert s.model.num_hidden_layers == 60
    assert s.model.steer_layer == 32
    # Submission shape tracks hidden_size.
    assert s.submission.expected_shape == (5376,)
    assert s.submission.expected_dtype == "float32"
    # Alpha bounds ordered.
    assert s.submission.alpha_min < s.submission.alpha_max


def test_all_concepts_present_in_registry_and_version_map():
    s = load_settings()
    allowed = set(s.concepts.active_allowed)
    assert allowed == set(DETECTOR_REGISTRY)
    assert allowed == set(s.detectors)


def test_steer_layer_within_range():
    s = load_settings()
    assert 0 <= s.model.steer_layer < s.model.num_hidden_layers


def test_invalid_config_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
model: {repo_id: x, revision: y, local_path: z, hidden_size: 100, num_hidden_layers: 10, steer_layer: 32}
quant: {load_in_4bit: true, bnb_4bit_quant_type: nf4, bnb_4bit_compute_dtype: bfloat16, bnb_4bit_use_double_quant: true}
submission: {tensor_name: direction, expected_shape: [100], expected_dtype: float32, norm_tolerance: 0.001, alpha_min: -1, alpha_max: 1}
generation: {max_new_tokens: 8, do_sample: false, seed: 1, batch_size: 2, padding_side: left, attn_implementation: eager}
prompts: {pool_path: p, pool_sha256_path: q, per_day: 10, pool_size: 100, dataset: d, dataset_revision: r}
concepts: {active_allowed: [birthday_cake]}
detectors: {birthday_cake: v1}
"""
    )
    # steer_layer 32 >= num_hidden_layers 10 -> invariant violation.
    with pytest.raises(ValueError):
        load_settings(str(bad))

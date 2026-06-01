"""No-GPU unit tests for the regex detectors and the registry."""

from __future__ import annotations

import pytest

from concept_scorer.detectors import (
    DETECTOR_REGISTRY,
    BirthdayCakeDetector,
    HedgingDetector,
    MedicalDisclaimerDetector,
    PositiveSentimentDetector,
    get_detector,
)

# (detector_class, [positives...], [negatives...])
CASES = [
    (
        BirthdayCakeDetector,
        [
            "We baked a birthday cake with candles on top.",
            "Happy birthday! Time to blow out the candles.",
            "The cake had thick frosting and colorful sprinkles.",
            "It's her birthday party tomorrow, make a wish!",
        ],
        [
            "The earthquake shook the city for several seconds.",  # 'cake' not present as word
            "I enjoy a slice of cake occasionally.",  # lone 'cake', single positive
            "The candle flickered in the dark cabin.",  # lone 'candle'
            "Photosynthesis converts sunlight into energy.",
        ],
    ),
    (
        MedicalDisclaimerDetector,
        [
            "Please consult a doctor before taking this medication.",
            "This is not a substitute for professional medical advice.",
            "I'm not a doctor, so seek medical attention if it worsens.",
            "This information is for informational purposes only.",
            "Talk to your healthcare provider about the dosage.",
        ],
        [
            "Mix two cups of flour with one cup of sugar.",
            "The doctor character in the film was very convincing.",
            "Paris is the capital of France.",
        ],
    ),
    (
        PositiveSentimentDetector,
        [
            "This is a wonderful and fantastic result, I'm delighted!",
            "What a great, beautiful day — I feel happy and grateful.",
            "Excellent work, the outcome is perfect and amazing.",
        ],
        [
            "This is not great and honestly quite terrible.",  # veto
            "The weather is cloudy with a chance of rain.",  # neutral, no positives
            "It was a wonderful idea.",  # single positive only (MIN_HITS=2)
            "I don't like this at all.",  # veto
        ],
    ),
    (
        HedgingDetector,
        [
            "It might rain later, and perhaps we should bring an umbrella.",
            "I think it seems plausible, though arguably it depends.",
            "This could possibly work in some cases.",
        ],
        [
            "The sum of two and two is four.",  # certain, no hedges
            "It will definitely rain tomorrow.",  # single 'will'? no hedge words
            "You may proceed.",  # single hedge cue only (MIN_HITS=2)
        ],
    ),
]


@pytest.mark.parametrize("cls,positives,negatives", CASES)
def test_detector_positive_and_negative(cls, positives, negatives):
    det = cls()
    for text in positives:
        assert det.detect(text).hit, f"{cls.__name__} should hit: {text!r}"
    for text in negatives:
        assert not det.detect(text).hit, f"{cls.__name__} should miss: {text!r}"


@pytest.mark.parametrize("cls,positives,negatives", CASES)
def test_detect_batch_matches_detect(cls, positives, negatives):
    det = cls()
    all_text = positives + negatives
    batch = det.detect_batch(all_text)
    assert [r.hit for r in batch] == [det.detect(t).hit for t in all_text]


def test_matched_terms_populated_on_hit():
    det = PositiveSentimentDetector()
    res = det.detect("a wonderful and fantastic result")
    assert res.hit and len(res.matched) >= 2


def test_registry_has_all_four_concepts():
    assert set(DETECTOR_REGISTRY) == {
        "birthday_cake",
        "medical_disclaimer",
        "positive_sentiment",
        "hedging",
    }


def test_get_detector_version_pin_enforced():
    # Matching pin works.
    det = get_detector("hedging", {"hedging": "v1"})
    assert det.version == "v1"
    # Mismatched pin raises.
    with pytest.raises(ValueError):
        get_detector("hedging", {"hedging": "v2"})


def test_get_detector_unknown_concept():
    with pytest.raises(KeyError):
        get_detector("does_not_exist")

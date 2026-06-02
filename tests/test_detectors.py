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
            "This is a wonderful and fantastic result, I'm delighted!",  # 4+4+3
            "What a great, beautiful day — I feel happy and grateful.",  # 3+3+3+3
            "Excellent work, the outcome is perfect and amazing.",  # 3+3+4
            "It was a wonderful idea.",  # 4: a single strong cue now hits (v1 missed it)
        ],
        [
            "The weather is cloudy with a chance of rain.",  # 0: no sentiment words
            "This is not great and honestly quite terrible.",  # not-great flipped + terrible -> -6
            "I feel sad and miserable about this awful outcome.",  # -2-3-3 = -8
            "I like it.",  # +2 < 3: below threshold
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
    det = get_detector("hedging", {"hedging": "v2"})
    assert det.version == "v2"
    # Mismatched pin raises.
    with pytest.raises(ValueError):
        get_detector("hedging", {"hedging": "v1"})


def test_get_detector_unknown_concept():
    with pytest.raises(KeyError):
        get_detector("does_not_exist")


def test_positive_sentiment_is_afinn_v3():
    det = PositiveSentimentDetector()
    assert det.version == "v3"
    # Net AFINN valence is surfaced as the continuous score.
    res = det.detect("a wonderful, fantastic outcome")  # 4 + 4
    assert res.hit and res.score == 8.0


def test_positive_sentiment_threshold_boundary():
    # Default threshold is 3.0: a lone +3 word hits (inclusive), a lone +2 word does not.
    det = PositiveSentimentDetector()
    assert det.detect("good").hit  # +3 == threshold
    assert not det.detect("like").hit  # +2 < 3


def test_positive_sentiment_threshold_is_configurable():
    strict = PositiveSentimentDetector(threshold=10.0)
    assert not strict.detect("a wonderful, fantastic outcome").hit  # 8 < 10
    lenient = PositiveSentimentDetector(threshold=2.0)
    assert lenient.detect("like").hit  # +2 >= 2


def test_get_detector_passes_threshold():
    # The scoring threshold reaches the constructor via get_detector.
    det = get_detector("positive_sentiment", {"positive_sentiment": "v3"}, threshold=100.0)
    assert not det.detect("a wonderful, fantastic, amazing, excellent day").hit  # 15 < 100


def test_weighted_regex_detector_scores_and_threshold():
    # Weighted lexicon: raw score is the summed cue weight; threshold decides the hit.
    d = BirthdayCakeDetector()  # DEFAULT_THRESHOLD = 2.0
    r = d.detect("We baked a birthday cake")
    assert r.hit and r.score == 5.0  # birthday cake 3 + cake 1 + birthday 1
    assert not d.detect("I enjoy a slice of cake").hit  # lone trapping 1.0 < 2.0
    assert d.detect("cake with candles").hit  # 1.0 + 1.0 >= 2.0
    # A lower threshold flips the lone-trapping result.
    assert BirthdayCakeDetector(threshold=1.0).detect("I enjoy a slice of cake").hit


def test_afinn_lexicon_sha256_guard():
    import hashlib

    from concept_scorer.detectors.afinn import AFINN_111_SHA256, _DATA_PATH, _load_lexicon

    # The pinned digest matches the vendored file, and a known entry loads.
    assert hashlib.sha256(_DATA_PATH.read_bytes()).hexdigest() == AFINN_111_SHA256
    lex = _load_lexicon(_DATA_PATH, AFINN_111_SHA256)
    assert lex["good"] == 3 and lex["terrible"] == -3
    # A wrong expected digest fails fast.
    with pytest.raises(ValueError):
        _load_lexicon(_DATA_PATH, "0" * 64)


def test_afinn_matches_multiword_phrases():
    from concept_scorer.detectors.afinn import score_text

    # AFINN's own multi-word entries score as the phrase, not the sum of their component words.
    assert score_text("not good")[0] == -2.0
    assert score_text("no fun")[0] == -3.0
    assert score_text("does not work")[0] == -3.0


def test_afinn_negation_window_flips_following_sentiment():
    from concept_scorer.detectors.afinn import score_text

    assert score_text("great")[0] == 3.0
    assert score_text("not great")[0] == -3.0  # "not great" is not an AFINN entry -> window flip
    assert score_text("isn't wonderful")[0] == -4.0  # n't contraction is a negator
    assert score_text("never excellent")[0] == -3.0


def test_positive_sentiment_rejects_negated_positives():
    det = PositiveSentimentDetector()  # default threshold 3.0
    # Negated/negative text must not register as positive (the code-review regression cases).
    assert not det.detect("This product is not good and not worth it.").hit
    assert not det.detect("That movie was not great at all.").hit
    assert not det.detect("No fun, just disappointment.").hit
    # Genuine positives still hit.
    assert det.detect("This is wonderful and fantastic.").hit

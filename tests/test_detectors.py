"""
Unit tests for detector implementations and common interface (FR-9 & FR-10).
"""

from src.detectors.base import BaseDetector
from src.detectors.self_consistency import SelfConsistencyDetector
from src.detectors.baselines import RandomGuesser, MajorityClassGuesser


def test_base_detector_interface():
    # Test that BaseDetector enforce interface
    detector = RandomGuesser(seed=42)
    assert isinstance(detector, BaseDetector)
    assert detector.name == "RandomGuesser"

    ctx = {"samples": ["Metformin starting dose is 500 mg."]}
    score = detector.score("Metformin starts at 500 mg.", ctx)
    assert 0.0 <= score <= 1.0

    pred = detector.predict("Metformin starts at 500 mg.", ctx, threshold=0.5)
    assert pred in (0, 1)


def test_self_consistency_detector_agreement():
    detector = SelfConsistencyDetector()
    claim = "Lisinopril initial dose is 10 mg once daily."
    context_supported = {
        "samples": [
            "Lisinopril is initiated at 10 mg daily for hypertension.",
            "The initial dose of lisinopril is 10 mg once daily.",
            "Adults start lisinopril at 10 mg once a day."
        ]
    }
    score_low = detector.score(claim, context_supported)
    # High agreement -> low hallucination score
    assert score_low <= 0.35

    context_divergent = {
        "samples": [
            "Lisinopril is not used for blood pressure.",
            "The initial dose is 100 mg twice daily.",
            "Atenolol is the drug of choice, not lisinopril."
        ]
    }
    score_high = detector.score(claim, context_divergent)
    # High disagreement -> high hallucination score
    assert score_high >= 0.65


def test_majority_guesser():
    detector = MajorityClassGuesser(majority_is_hallucinated=False)
    pred = detector.predict("Any claim", {})
    assert pred == 0


"""
Unit tests for all 4 detection modules and score fusion engine.
"""

from backend.detection.eigenscore import EigenScoreDetector
from backend.detection.self_consistency import SelfConsistencyDetector
from backend.detection.semantic_entropy import SemanticEntropyDetector
from backend.detection.retrieval_verification import RetrievalVerificationDetector
from backend.detection.score_fusion import MultiSignalScoreFusion


def test_eigenscore_spectrum_and_scoring():
    detector = EigenScoreDetector()
    # Test factual simulation
    score_factual = detector.score("Metformin is started at 500 mg daily.", context={"is_hallucinated": False, "seed": 42})
    # Test hallucinated simulation
    score_hallucinated = detector.score("Dr. Henderson invented a 450 mg protocol.", context={"is_hallucinated": True, "seed": 42})

    assert 0.0 <= score_factual <= 1.0
    assert 0.0 <= score_hallucinated <= 1.0
    assert score_hallucinated > score_factual


def test_self_consistency_consensus():
    detector = SelfConsistencyDetector()
    claim = "Lisinopril starting dose is 10 mg."
    samples_agree = ["Initial lisinopril dose is 10 mg.", "Start with 10 mg of lisinopril."]
    samples_disagree = ["Lisinopril is not used for blood pressure.", "Starting dose is 150 mg."]

    score_agree = detector.score(claim, context={"samples": samples_agree})
    score_disagree = detector.score(claim, context={"samples": samples_disagree})

    assert score_agree < score_disagree
    assert score_agree <= 0.35


def test_semantic_entropy_clustering():
    detector = SemanticEntropyDetector()
    # Semantically identical samples (different wording)
    identical_samples = [
        "The capital of Australia is Canberra.",
        "Canberra serves as the capital city of Australia.",
        "Australia's federal capital is Canberra."
    ]
    norm_ent, raw_ent, clusters = detector.calculate_entropy(identical_samples)
    assert len(clusters) == 1
    assert norm_ent == 0.0  # Zero semantic entropy

    # Conflicting samples
    conflicting_samples = [
        "The capital of Australia is Canberra.",
        "The capital of Australia is Sydney.",
        "Melbourne is the capital of Australia."
    ]
    norm_ent_conf, _, clusters_conf = detector.calculate_entropy(conflicting_samples)
    assert len(clusters_conf) >= 2
    assert norm_ent_conf > 0.5  # High semantic entropy


def test_retrieval_verification_healthcare():
    detector = RetrievalVerificationDetector()
    supported_claim = "The recommended starting dose of metformin is 500 mg orally twice daily."
    res = detector.verify_claim(supported_claim, domain="healthcare")
    assert res["classification"] == "Supported"
    assert res["hallucination_score"] <= 0.20

    contradicted_claim = "The recommended starting dose of metformin is 9000 mg orally twice daily."
    res_contra = detector.verify_claim(contradicted_claim, domain="healthcare")
    assert res_contra["classification"] == "Contradicted"
    assert res_contra["hallucination_score"] >= 0.85


def test_score_fusion_engine():
    fusion = MultiSignalScoreFusion()
    signals_consistent = {
        "eigenscore": 0.20,
        "self_consistency": 0.15,
        "semantic_entropy": 0.10,
        "retrieval": 0.10
    }
    res_low = fusion.fuse_signals(signals_consistent)
    assert res_low["final_score"] < 0.35
    assert "Low Risk" in res_low["risk_level"]

    signals_hallucinated = {
        "eigenscore": 0.85,
        "self_consistency": 0.90,
        "semantic_entropy": 0.88,
        "retrieval": 0.95
    }
    res_high = fusion.fuse_signals(signals_hallucinated)
    assert res_high["final_score"] > 0.65
    assert "Elevated Risk" in res_high["risk_level"]


import pytest
from app.detection.claim_extractor import claim_extractor
from app.detection.self_consistency import self_consistency_analyzer
from app.detection.retrieval import retrieval_verifier
from app.detection.uncertainty import uncertainty_estimator
from app.detection.fusion import fusion_engine
from app.dataset.benchmark_manager import benchmark_manager
from app.evaluation.experiment_runner import experiment_runner

def test_claim_extractor():
    text = "Metformin was approved by the FDA in 1994. It treats type 2 diabetes."
    claims = claim_extractor.extract_claims(text)
    assert len(claims) >= 1
    assert "claim_id" in claims[0]
    assert "claim" in claims[0]

def test_self_consistency():
    claims = [{"claim_id": "C001", "claim": "Metformin was approved in 1994."}]
    alt_gens = ["Metformin approved in 1994.", "Metformin was approved in 2012."]
    res = self_consistency_analyzer.analyze_consistency(claims, alt_gens)
    assert "overall_consistency_score" in res
    assert "inconsistency_risk" in res

def test_retrieval_verifier():
    claims = [{"claim_id": "C001", "claim": "Metformin was approved by the FDA in 1994 for type 2 diabetes."}]
    corpus = benchmark_manager.get_domain_corpus("healthcare")
    res = retrieval_verifier.verify_claims(claims, corpus)
    assert "overall_retrieval_risk" in res
    assert len(res["claim_verifications"]) > 0

def test_uncertainty_estimator():
    res = uncertainty_estimator.estimate_uncertainty("Sample response text")
    assert "overall_uncertainty_score" in res
    assert "semantic_entropy" in res

def test_fusion_engine():
    claims = [{"claim_id": "C001", "claim": "Metformin was approved in 1994."}]
    sc_res = {"inconsistency_risk": 0.2, "claim_consistency": [{"claim_id": "C001", "inconsistency_risk": 0.2, "conflicts": 0}]}
    ret_res = {"overall_retrieval_risk": 0.05, "claim_verifications": [{"claim_id": "C001", "status": "SUPPORTED", "retrieval_risk": 0.05}]}
    unc_res = {"overall_uncertainty_score": 0.15}
    
    fusion = fusion_engine.fuse_detection_signals(claims, sc_res, ret_res, unc_res)
    assert "overall_hallucination_risk" in fusion
    assert "final_verdict" in fusion

def test_experiment_runner():
    res = experiment_runner.run_benchmark_matrix(domain="healthcare")
    assert "experiment_results" in res
    assert "E1" in res["experiment_results"]
    assert "E7" in res["experiment_results"]

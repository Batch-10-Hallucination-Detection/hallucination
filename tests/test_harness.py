"""
Unit tests for Evaluation Harness and Metrics calculation (FR-11).
"""

from src.schema import BenchmarkRow, ClaimLabel
from src.evaluation.metrics import compute_binary_metrics, find_optimal_threshold
from src.evaluation.harness import EvaluationHarness
from src.detectors.baselines import RandomGuesser


def test_metrics_calculation():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]

    metrics = compute_binary_metrics(y_true, y_scores, threshold=0.5)
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0


def test_optimal_threshold_finding():
    y_true = [0, 0, 1, 1]
    y_scores = [0.2, 0.4, 0.6, 0.8]

    best_thresh, best_f1 = find_optimal_threshold(y_true, y_scores)
    assert best_f1 == 1.0
    assert 0.4 < best_thresh <= 0.6


def test_harness_evaluation_flow(tmp_path):
    harness = EvaluationHarness(output_dir=str(tmp_path))
    detector = RandomGuesser(seed=123)

    rows = [
        BenchmarkRow(
            prompt_id="PROMPT-01",
            prompt="Q1",
            evidence=["E1"],
            model="test-model",
            response="Ans1",
            samples=["S1", "S2"],
            claim_id="C1",
            claim="Claim 1",
            label=ClaimLabel.SUPPORTED,
            annotator_ids=["a1"]
        ),
        BenchmarkRow(
            prompt_id="PROMPT-02",
            prompt="Q2",
            evidence=["E2"],
            model="test-model",
            response="Ans2",
            samples=["S1", "S2"],
            claim_id="C2",
            claim="Claim 2",
            label=ClaimLabel.CONTRADICTED,
            annotator_ids=["a1"]
        )
    ]

    res = harness.evaluate_detector(detector, rows, split_name="test", threshold=0.5)
    assert "summary" in res
    assert res["summary"].total_claims == 2
    assert res["summary"].detector_name == "RandomGuesser"


"""
Error Analysis and Diagnostic Export Module (FR-12).
Exports False Positives and False Negatives with claims, evidence, scores, and failure taxonomy.
Fulfills FR-12: 'List of false positives and false negatives with claim, evidence and score'.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.schema import BenchmarkRow, DetectorScoreResult


class ErrorAnalyzer:
    """
    Analyzes and exports detector error distributions.
    """

    def __init__(self, output_dir: str = "outputs/error_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_and_export(
        self,
        benchmark_rows: List[BenchmarkRow],
        predictions: List[DetectorScoreResult],
        detector_name: str = "SelfConsistency"
    ) -> Dict[str, Any]:
        """
        Classifies errors into False Positives and False Negatives,
        exports CSVs, and produces a structured failure pattern summary.
        """
        row_map = {r.claim_id: r for r in benchmark_rows}

        false_positives: List[Dict[str, Any]] = []
        false_negatives: List[Dict[str, Any]] = []

        for p in predictions:
            row = row_map.get(p.claim_id)
            if not row:
                continue

            ground_truth = row.is_hallucinated
            pred = p.prediction

            # False Positive: Model flagged hallucination, but it was actually Supported
            if pred == 1 and ground_truth == 0:
                failure_cat = "Lexical divergence in samples despite factual validity"
                if len(row.samples) > 0 and any("guideline" in s.lower() for s in row.samples):
                    failure_cat = "Paraphrase variation across sample phrasing"

                false_positives.append({
                    "claim_id": row.claim_id,
                    "prompt_id": row.prompt_id,
                    "prompt": row.prompt,
                    "claim": row.claim,
                    "ground_truth_label": row.label.value,
                    "detector_score": p.score,
                    "threshold": p.threshold_used,
                    "failure_category": failure_cat,
                    "evidence_snippet": row.evidence[0][:150] if row.evidence else "",
                    "sample_snippet": row.samples[0][:150] if row.samples else ""
                })

            # False Negative: Model marked supported, but it was actually Hallucinated
            elif pred == 0 and ground_truth == 1:
                failure_cat = "Consistent hallucination across sampled responses"
                if any(num in row.claim for num in ["mg", "dose", "450", "daily"]):
                    failure_cat = "Subtle numerical/dosage error repeated across stochastic samples"

                false_negatives.append({
                    "claim_id": row.claim_id,
                    "prompt_id": row.prompt_id,
                    "prompt": row.prompt,
                    "claim": row.claim,
                    "ground_truth_label": row.label.value,
                    "detector_score": p.score,
                    "threshold": p.threshold_used,
                    "failure_category": failure_cat,
                    "evidence_snippet": row.evidence[0][:150] if row.evidence else "",
                    "sample_snippet": row.samples[0][:150] if row.samples else ""
                })

        # Save to CSV
        fp_path = self.output_dir / "false_positives.csv"
        fn_path = self.output_dir / "false_negatives.csv"

        df_fp = pd.DataFrame(false_positives)
        df_fn = pd.DataFrame(false_negatives)

        df_fp.to_csv(fp_path, index=False)
        df_fn.to_csv(fn_path, index=False)

        print(f"[ErrorAnalysis] Saved {len(false_positives)} False Positives to {fp_path}")
        print(f"[ErrorAnalysis] Saved {len(false_negatives)} False Negatives to {fn_path}")

        return {
            "total_errors": len(false_positives) + len(false_negatives),
            "num_false_positives": len(false_positives),
            "num_false_negatives": len(false_negatives),
            "false_positives_file": str(fp_path),
            "false_negatives_file": str(fn_path)
        }


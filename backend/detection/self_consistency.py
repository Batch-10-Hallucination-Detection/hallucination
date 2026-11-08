"""
Self-Consistency Hallucination Detection Module.
Generates multiple answer samples for the same prompt and identifies factual disagreements between them.
Fulfills Detection Engine Layer 4 (Method B).
"""

import re
from typing import Dict, Any, List, Optional


class SelfConsistencyDetector:
    """
    Self-Consistency Detector:
    Scores claims or responses based on consensus across N stochastic samples.
    Claims with high divergence across samples receive elevated hallucination scores.
    """

    def __init__(self, nli_threshold: float = 0.55):
        self.name = "Self-Consistency"
        self.nli_threshold = nli_threshold

    def _entailment_similarity(self, premise: str, hypothesis: str) -> float:
        """
        Calculates directional entailment score between premise sample and hypothesis claim.
        Incorporates entity overlap, numeric consistency, and negation checks.
        """
        p_lower = premise.lower()
        h_lower = hypothesis.lower()

        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to", "for", "with", "that", "this", "it"}
        h_tokens = [w for w in re.findall(r"\b\w+\b", h_lower) if w not in stop_words]

        if not h_tokens:
            return 1.0

        matches = sum(1 for token in h_tokens if token in p_lower)
        overlap = matches / len(h_tokens)

        # Numerical & dosage mismatch penalty
        h_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", h_lower))
        p_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", p_lower))
        if h_nums and not h_nums.issubset(p_nums):
            overlap *= 0.35

        # Negation check
        neg_words = {"not", "never", "no", "contraindicated", "prohibited", "avoid", "cannot"}
        has_h_neg = any(w in h_lower for w in neg_words)
        has_p_neg = any(w in p_lower for w in neg_words)
        if has_h_neg != has_p_neg:
            overlap *= 0.2

        return min(max(overlap, 0.0), 1.0)

    def score(self, claim: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Returns continuous hallucination score in [0.0, 1.0].
        Score = fraction of samples that fail to entail the claim.
        """
        context = context or {}
        samples: List[str] = context.get("samples", [])

        if not samples:
            return 0.5

        non_support_count = 0.0
        for sample in samples:
            sim = self._entailment_similarity(premise=sample, hypothesis=claim)
            if sim < self.nli_threshold:
                non_support_count += 1.0

        score_val = non_support_count / len(samples)
        return round(float(score_val), 4)

    def analyze_claim(self, claim: str, samples: List[str]) -> Dict[str, Any]:
        """Detailed analysis including sample-level consistency breakdown."""
        evaluations = []
        supports = 0
        for idx, sample in enumerate(samples):
            entail = self._entailment_similarity(premise=sample, hypothesis=claim)
            is_supported = entail >= self.nli_threshold
            if is_supported:
                supports += 1
            evaluations.append({
                "sample_id": idx + 1,
                "entailment_score": round(entail, 4),
                "is_supported": is_supported,
                "text_snippet": sample[:100] + "..." if len(sample) > 100 else sample
            })

        score_val = (len(samples) - supports) / max(len(samples), 1)
        return {
            "score": round(score_val, 4),
            "total_samples": len(samples),
            "supporting_samples": supports,
            "dissenting_samples": len(samples) - supports,
            "sample_evaluations": evaluations
        }


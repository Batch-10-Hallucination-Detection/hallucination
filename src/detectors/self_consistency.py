"""
Self-Consistency Hallucination Detector (FR-9).
Implements SelfCheckGPT-style NLI consistency detection across N stochastic samples.
Scores claims by agreement across samples: score = share of samples that do not support the claim.
"""

import re
from typing import Dict, Any, List, Optional
from src.detectors.base import BaseDetector


class SelfConsistencyDetector(BaseDetector):
    """
    Self-Consistency Detector fulfilling FR-9 and PRD Section 7.
    Evaluates atomic claims against N sampled responses using NLI logic:
    Score = Fraction of sampled responses that fail to entail the claim.
    A score near 1.0 indicates strong divergence across samples (hallucination).
    A score near 0.0 indicates strong consensus across samples (truthful/supported).
    """

    def __init__(
        self,
        nli_model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 16
    ):
        self._name = "SelfConsistencyDetector"
        self.nli_model_name = nli_model_name
        self.device = device
        self.batch_size = batch_size
        self._nli_model = None
        self._init_model()

    @property
    def name(self) -> str:
        return self._name

    def _init_model(self):
        """Attempts to load a neural NLI model if specified and available."""
        if self.nli_model_name:
            try:
                from transformers import pipeline
                self._nli_model = pipeline(
                    "text-classification",
                    model=self.nli_model_name,
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as e:
                # Graceful fallback to heuristic semantic NLI matcher
                self._nli_model = None

    def _evaluate_pair_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Evaluates whether premise entails hypothesis.
        Returns entailment score between 0.0 (contradiction/unrelated) and 1.0 (entailed).
        """
        if self._nli_model:
            try:
                # Standard NLI input format: premise and hypothesis
                res = self._nli_model({"text": premise, "text_pair": hypothesis})
                label = res["label"].lower()
                score = res["score"]
                if "entail" in label:
                    return score
                elif "contradict" in label:
                    return 0.0
                else:  # neutral
                    return score * 0.3
            except Exception:
                pass

        # Heuristic semantic / entity overlap NLI matcher (fast, robust, offline)
        premise_lower = premise.lower()
        hypo_lower = hypothesis.lower()

        # Clean words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to", "for", "with", "that", "this", "it"}
        hypo_tokens = [w for w in re.findall(r"\b\w+\b", hypo_lower) if w not in stop_words]

        if not hypo_tokens:
            return 1.0

        matches = sum(1 for token in hypo_tokens if token in premise_lower)
        overlap_ratio = matches / len(hypo_tokens)

        # Check for numeric or dosage conflicts
        hypo_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", hypo_lower))
        premise_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", premise_lower))

        if hypo_numbers and not hypo_numbers.issubset(premise_numbers):
            # Numeric claim contains numbers not found in premise -> lower entailment
            overlap_ratio *= 0.4

        # Check for negation mismatch
        has_hypo_neg = any(neg in hypo_lower for neg in ["not", "never", "contraindicated", "prohibited", "avoid"])
        has_premise_neg = any(neg in premise_lower for neg in ["not", "never", "contraindicated", "prohibited", "avoid"])
        if has_hypo_neg != has_premise_neg:
            overlap_ratio *= 0.2

        return min(max(overlap_ratio, 0.0), 1.0)

    def score(self, claim: str, context: Dict[str, Any]) -> float:
        """
        Scores hallucination likelihood across N samples.
        context must contain 'samples' (List[str]).
        """
        samples = context.get("samples", [])
        if not samples:
            # If no samples provided, check against main response or default to uncertainty (0.5)
            response = context.get("response", "")
            if response:
                entail_score = self._evaluate_pair_entailment(premise=response, hypothesis=claim)
                return 1.0 - entail_score
            return 0.5

        non_support_count = 0.0
        for sample in samples:
            entail_score = self._evaluate_pair_entailment(premise=sample, hypothesis=claim)
            # A sample supports the claim if entailment score >= 0.55
            if entail_score < 0.55:
                non_support_count += 1.0

        # Share of samples that do NOT support the claim
        hallucination_score = non_support_count / len(samples)
        return round(float(hallucination_score), 4)

    def score_with_details(self, claim: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the hallucination score alongside per-sample agreement breakdown."""
        samples = context.get("samples", [])
        sample_scores = []
        for idx, sample in enumerate(samples):
            entail_score = self._evaluate_pair_entailment(premise=sample, hypothesis=claim)
            sample_scores.append({
                "sample_idx": idx,
                "entailment_score": round(entail_score, 4),
                "supports": entail_score >= 0.55,
                "sample_snippet": sample[:80] + "..." if len(sample) > 80 else sample
            })

        overall_score = self.score(claim, context)
        return {
            "score": overall_score,
            "num_samples": len(samples),
            "samples_eval": sample_scores
        }


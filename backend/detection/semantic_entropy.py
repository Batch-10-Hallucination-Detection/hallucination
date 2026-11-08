"""
Semantic Entropy Hallucination Detection Module.
Measures semantic uncertainty across generated answers by clustering semantically equivalent
responses via bidirectional entailment / semantic similarity rather than treating lexical
variations as distinct meanings (Kuhn et al. 2023, Farquhar et al. 2024).
Fulfills Detection Engine Layer 4 (Method C).
"""

import math
import re
from typing import Dict, Any, List, Optional, Tuple


class SemanticEntropyDetector:
    """
    Semantic Entropy Detector:
    Clusters N stochastic answers into K semantic equivalence classes.
    Computes Shannon entropy over semantic classes: H_semantic = - sum(p_k * log2(p_k)).
    High semantic entropy indicates model uncertainty across conflicting meanings (hallucination).
    """

    def __init__(self, clustering_threshold: float = 0.65):
        self.name = "Semantic-Entropy"
        self.clustering_threshold = clustering_threshold

    def _are_semantically_equivalent(self, text_a: str, text_b: str) -> bool:
        """
        Determines if text_a and text_b express the same semantic meaning
        (bidirectional equivalence) despite lexical differences.
        """
        a_clean = text_a.lower().strip()
        b_clean = text_b.lower().strip()

        if a_clean == b_clean:
            return True

        # Extract content words (nouns, numbers, clinical terms)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to", "for", "with", "that", "this", "it", "and", "or"}
        tokens_a = set(w for w in re.findall(r"\b\w+\b", a_clean) if w not in stop_words)
        tokens_b = set(w for w in re.findall(r"\b\w+\b", b_clean) if w not in stop_words)

        if not tokens_a or not tokens_b:
            return False

        # Overlap coefficient (containment) and Jaccard
        intersection = len(tokens_a & tokens_b)
        min_len = min(len(tokens_a), len(tokens_b))
        overlap_coeff = intersection / min_len if min_len > 0 else 0.0

        # Named entity / distinct token conflict check:
        # If both statements introduce distinct, unshared entity tokens, they express conflicting claims
        diff_a = tokens_a - tokens_b
        diff_b = tokens_b - tokens_a
        if diff_a and diff_b:
            # Check if diff contains substantive entity words (e.g., 'canberra' vs 'sydney')
            generic_fillers = {"serves", "city", "federal", "stated", "known", "called", "primarily", "initial", "generally"}
            substantive_a = [w for w in diff_a if w not in generic_fillers]
            substantive_b = [w for w in diff_b if w not in generic_fillers]
            if substantive_a and substantive_b:
                return False

        # Numerical conflict check: numbers must match exactly if present
        nums_a = set(re.findall(r"\b\d+(?:\.\d+)?\b", a_clean))
        nums_b = set(re.findall(r"\b\d+(?:\.\d+)?\b", b_clean))
        if nums_a and nums_b and nums_a != nums_b:
            return False

        # Negation conflict check
        neg_words = {"not", "never", "no", "contraindicated", "prohibited", "avoid"}
        neg_a = bool(tokens_a & neg_words)
        neg_b = bool(tokens_b & neg_words)
        if neg_a != neg_b:
            return False

        return overlap_coeff >= self.clustering_threshold

    def cluster_responses(self, samples: List[str]) -> List[List[str]]:
        """
        Clusters sampled responses into semantic equivalence classes.
        """
        clusters: List[List[str]] = []

        for sample in samples:
            assigned = False
            for cluster in clusters:
                # Compare with cluster representative (first sample in cluster)
                if self._are_semantically_equivalent(sample, cluster[0]):
                    cluster.append(sample)
                    assigned = True
                    break
            if not assigned:
                clusters.append([sample])

        return clusters

    def calculate_entropy(self, samples: List[str]) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Calculates raw semantic entropy and normalized entropy in [0, 1].
        Returns:
            normalized_entropy: float in [0.0, 1.0]
            raw_entropy_bits: float
            cluster_details: list of cluster metadata
        """
        N = len(samples)
        if N <= 1:
            return 0.0, 0.0, [{"cluster_id": 1, "size": N, "probability": 1.0, "representative": samples[0] if samples else ""}]

        clusters = self.cluster_responses(samples)
        raw_entropy = 0.0
        cluster_details = []

        for idx, cluster in enumerate(clusters):
            prob = len(cluster) / float(N)
            raw_entropy -= prob * math.log2(prob + 1e-12)
            cluster_details.append({
                "cluster_id": idx + 1,
                "size": len(cluster),
                "probability": round(prob, 4),
                "representative": cluster[0][:100] + "..." if len(cluster[0]) > 100 else cluster[0],
                "sample_count": len(cluster)
            })

        max_entropy = math.log2(N)
        normalized_entropy = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0
        normalized_entropy = min(max(normalized_entropy, 0.0), 1.0)

        return round(normalized_entropy, 4), round(raw_entropy, 4), cluster_details

    def score(self, text: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Standard scoring interface: returns normalized semantic entropy.
        """
        context = context or {}
        samples = context.get("samples", [])
        if not samples:
            return 0.5

        norm_entropy, _, _ = self.calculate_entropy(samples)
        return norm_entropy

    def analyze_semantic_uncertainty(self, samples: List[str]) -> Dict[str, Any]:
        """Provides full diagnostic report on semantic clustering and entropy."""
        norm_entropy, raw_entropy, cluster_details = self.calculate_entropy(samples)
        return {
            "score": norm_entropy,
            "raw_entropy_bits": raw_entropy,
            "num_clusters": len(cluster_details),
            "num_samples": len(samples),
            "is_semantically_consistent": len(cluster_details) == 1,
            "clusters": cluster_details
        }


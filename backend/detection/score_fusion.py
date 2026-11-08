"""
Multi-Signal Score Fusion and Explanation Module.
Calibrates and fuses detection signals (EigenScore, Self-Consistency, Semantic Entropy, Retrieval Verification)
to produce explainable claim-level risk assessments and attribution.
Fulfills Detection Engine Layer 5.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class MultiSignalScoreFusion:
    """
    Score Fusion Engine:
    Combines internal-state representation analysis (EigenScore),
    stochastic consensus (Self-Consistency), semantic uncertainty (Semantic Entropy),
    and external grounding (Retrieval Verification).
    """

    DEFAULT_WEIGHTS = {
        "eigenscore": 0.25,
        "self_consistency": 0.30,
        "semantic_entropy": 0.25,
        "retrieval": 0.20
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._normalize_weights()

    def _normalize_weights(self):
        """Ensures weights sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    @staticmethod
    def calibrate_signal(raw_score: float, signal_name: str) -> float:
        """
        Calibrates raw detector output to a normalized probabilistic scale [0.0, 1.0]
        using empirical sigmoid transfer functions.
        """
        score = min(max(float(raw_score), 0.0), 1.0)
        if signal_name == "semantic_entropy":
            # Semantic entropy has high dynamic range near 0.5-0.8
            return float(1.0 / (1.0 + np.exp(-6.0 * (score - 0.45))))
        elif signal_name == "eigenscore":
            # EigenScore representation dispersion
            return float(1.0 / (1.0 + np.exp(-5.0 * (score - 0.50))))
        return score

    def fuse_signals(
        self,
        signals: Dict[str, float],
        active_methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fuses active detection signals using normalized calibrated combination.
        signals: dict with keys matching ['eigenscore', 'self_consistency', 'semantic_entropy', 'retrieval']
        """
        active_keys = active_methods or list(signals.keys())
        active_weights = {k: self.weights.get(k, 0.25) for k in active_keys if k in signals}

        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            return {"final_score": 0.5, "risk_level": "Moderate Risk", "attribution": {}}

        normalized_active_weights = {k: v / total_weight for k, v in active_weights.items()}

        calibrated_signals = {}
        weighted_sum = 0.0
        attributions = {}

        for k, weight in normalized_active_weights.items():
            raw_val = signals[k]
            cal_val = self.calibrate_signal(raw_val, k)
            calibrated_signals[k] = round(cal_val, 4)
            contribution = weight * cal_val
            weighted_sum += contribution
            attributions[k] = {
                "raw_score": round(raw_val, 4),
                "calibrated_score": round(cal_val, 4),
                "weight": round(weight, 4),
                "contribution": round(contribution, 4)
            }

        final_score = round(float(np.clip(weighted_sum, 0.0, 1.0)), 4)

        # Risk classification
        if final_score < 0.35:
            risk_level = "Low Risk (Likely Truthful)"
            badge_color = "green"
        elif final_score < 0.65:
            risk_level = "Moderate Risk (Needs Verification)"
            badge_color = "yellow"
        else:
            risk_level = "Elevated Risk (Potential Hallucination)"
            badge_color = "red"

        # Generate human-readable explanation
        explanation = self.generate_explanation(signals, final_score, risk_level)

        return {
            "final_score": final_score,
            "risk_level": risk_level,
            "badge_color": badge_color,
            "calibrated_signals": calibrated_signals,
            "attribution": attributions,
            "explanation": explanation
        }

    def generate_explanation(self, signals: Dict[str, float], final_score: float, risk_level: str) -> str:
        """Synthesizes qualitative explanation of the multi-signal findings."""
        reasons = []

        if signals.get("retrieval", 0) >= 0.70:
            reasons.append("Retrieved trusted documents directly contradict or fail to corroborate the assertion")
        if signals.get("self_consistency", 0) >= 0.60:
            reasons.append("Multiple generated samples disagree on key factual elements")
        if signals.get("semantic_entropy", 0) >= 0.60:
            reasons.append("High semantic entropy indicates model uncertainty across conflicting answer meanings")
        if signals.get("eigenscore", 0) >= 0.65:
            reasons.append("Internal representation hidden states exhibit high dispersion / spectral degeneration")

        if not reasons:
            if final_score < 0.35:
                return "All signals confirm strong consensus across model representations, sampled answers, and reference evidence."
            return "Signals indicate moderate consistency with no critical contradictions detected."

        return "; ".join(reasons) + "."


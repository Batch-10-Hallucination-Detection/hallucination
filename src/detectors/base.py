"""
Base Detector Interface (FR-10).
Defines the standard abstract plugin interface that every hallucination detector must implement
to run inside the evaluation harness.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.schema import DetectorScoreResult


class BaseDetector(ABC):
    """
    Common detector interface fulfilling FR-10:
    'Any detector that implements one scoring function runs in the harness'.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the detector."""
        pass

    @abstractmethod
    def score(self, claim: str, context: Dict[str, Any]) -> float:
        """
        Calculates a continuous hallucination score between 0.0 and 1.0.
        0.0 = completely supported / consistent (not hallucinated)
        1.0 = completely uncorroborated / contradicted (hallucinated)

        Parameters:
            claim: The atomic claim text being evaluated.
            context: Dictionary containing prompt, response, samples, evidence, etc.
        """
        pass

    def predict(self, claim: str, context: Dict[str, Any], threshold: float = 0.5) -> int:
        """
        Converts the continuous score into a binary decision (0 = Not Hallucinated, 1 = Hallucinated).
        """
        return 1 if self.score(claim, context) >= threshold else 0

    def score_batch(self, claims: List[str], contexts: List[Dict[str, Any]]) -> List[float]:
        """Scores a batch of claims."""
        return [self.score(c, ctx) for c, ctx in zip(claims, contexts)]


"""
Baseline Detectors (FR-11).
Implements RandomGuesser and MajorityClassGuesser to establish empirical baseline floors.
"""

import random
from typing import Dict, Any, Optional
from src.detectors.base import BaseDetector


class RandomGuesser(BaseDetector):
    """
    Uniform random baseline detector.
    Outputs uniformly distributed scores in [0.0, 1.0].
    """

    def __init__(self, seed: int = 42):
        self._name = "RandomGuesser"
        self.seed = seed
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self._name

    def score(self, claim: str, context: Dict[str, Any]) -> float:
        return round(self.rng.random(), 4)


class MajorityClassGuesser(BaseDetector):
    """
    Majority class baseline detector.
    Always predicts the majority class observed on the training/dev distribution.
    If majority is Supported (0), predicts 0.0 score.
    If majority is Hallucinated (1), predicts 1.0 score.
    """

    def __init__(self, majority_is_hallucinated: bool = False, prior_prob: float = 0.30):
        self._name = "MajorityClassGuesser"
        self.majority_is_hallucinated = majority_is_hallucinated
        self.prior_prob = prior_prob

    @property
    def name(self) -> str:
        return self._name

    def score(self, claim: str, context: Dict[str, Any]) -> float:
        # Returns prior probability of hallucination as confidence
        return self.prior_prob

    def predict(self, claim: str, context: Dict[str, Any], threshold: float = 0.5) -> int:
        return 1 if self.majority_is_hallucinated else 0


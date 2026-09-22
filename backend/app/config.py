import os
from pydantic import BaseModel

class AppConfig(BaseModel):
    APP_NAME: str = "Hallucination Detection Framework"
    VERSION: str = "1.0.0"
    DEFAULT_DOMAIN: str = "healthcare"
    
    # Default Fusion Weights: w1 (Self-Consistency), w2 (Retrieval), w3 (Uncertainty)
    WEIGHT_SELF_CONSISTENCY: float = 0.30
    WEIGHT_RETRIEVAL: float = 0.50
    WEIGHT_UNCERTAINTY: float = 0.20
    
    # Risk Classification Thresholds
    HIGH_RISK_THRESHOLD: float = 0.65
    MEDIUM_RISK_THRESHOLD: float = 0.35

    # Verification Thresholds
    RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.45
    NLI_CONTRADICTION_THRESHOLD: float = 0.60
    
config = AppConfig()

"""
Core Schema and Data Models for Hallucination Detection Framework (Phase 1).
Fulfills FR-1, FR-2, FR-7, FR-8, and PRD Table 6.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ClaimLabel(str, Enum):
    """3-way label taxonomy for medical claims."""
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    UNVERIFIABLE = "Unverifiable"


def label_to_binary(label: ClaimLabel | str) -> int:
    """
    Binary mapping per PRD Section 5:
    Supported = 0 (Not hallucinated)
    Contradicted = 1 (Hallucinated)
    Unverifiable = 1 (Hallucinated)
    """
    if isinstance(label, str):
        label_val = label.strip()
    else:
        label_val = label.value

    if label_val.lower() == "supported":
        return 0
    elif label_val.lower() in ("contradicted", "unverifiable"):
        return 1
    else:
        raise ValueError(f"Unknown claim label: {label}")


class PromptItem(BaseModel):
    """Domain prompt representation (FR-1)."""
    id: str = Field(..., description="Unique prompt ID, e.g., 'PROMPT-001'")
    question: str = Field(..., description="Domain medical question")
    source: str = Field(..., description="Public origin benchmark (e.g., Med-HALT, PubMedQA, HaluBench)")
    topic: str = Field(..., description="Topic category (e.g., Drug Dosage, Contraindications)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional source metadata")


class CorpusDocument(BaseModel):
    """Reference corpus document representation (FR-2)."""
    doc_id: str = Field(..., description="Unique document ID (e.g., 'DOC-PUBMED-1234')")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Passage or document body")
    source_url: str = Field(..., description="Trusted source URL or DOI")
    license: str = Field(..., description="Public open license (e.g., CC-BY 4.0, Open Access)")
    date: str = Field(..., description="Publication or accession date (YYYY-MM-DD)")
    topic: str = Field(..., description="Medical topic")


class AtomicClaim(BaseModel):
    """Single checkable claim extracted from an LLM response (FR-5)."""
    claim_id: str = Field(..., description="Unique claim ID, e.g., 'PROMPT-001-C01'")
    claim: str = Field(..., description="Atomic factual statement")
    source_sentence: Optional[str] = Field(None, description="Original sentence containing the claim")
    char_start: Optional[int] = Field(None, description="Start character offset")
    char_end: Optional[int] = Field(None, description="End character offset")


class BenchmarkRow(BaseModel):
    """
    Unified Benchmark Row Specification (v0.1) fulfilling PRD Section 6 (FR-8).
    Each row represents one claim linked to prompt, evidence, and response context.
    """
    prompt_id: str = Field(..., description="Unique ID of the question")
    prompt: str = Field(..., description="Domain question given to the LLM")
    evidence: List[str] = Field(default_factory=list, description="Retrieved passages from trusted sources")
    model: str = Field(..., description="LLM name and version (e.g., 'Llama-3-8B-Instruct')")
    response: str = Field(..., description="Main generated answer")
    samples: List[str] = Field(default_factory=list, description="N extra sampled answers for consistency checks")
    claim_id: str = Field(..., description="Unique claim ID")
    claim: str = Field(..., description="One atomic claim from the response")
    label: ClaimLabel = Field(..., description="Supported, Contradicted, or Unverifiable")
    annotator_ids: List[str] = Field(default_factory=lambda: ["annotator_1"], description="Annotators who labeled it")
    notes: Optional[str] = Field(None, description="Optional annotation rationale or clinical notes")
    topic: Optional[str] = Field(None, description="Medical topic category")
    seed: Optional[int] = Field(42, description="Random seed used for generation")

    @property
    def is_hallucinated(self) -> int:
        """Returns 1 if hallucinated (Contradicted or Unverifiable), 0 if Supported."""
        return label_to_binary(self.label)


class AnnotationRecord(BaseModel):
    """Record used for inter-annotator agreement calculations (FR-7)."""
    claim_id: str
    prompt_id: str
    annotator_id: str
    label: ClaimLabel
    notes: Optional[str] = None


class DetectorScoreResult(BaseModel):
    """Result of scoring an individual claim."""
    claim_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Hallucination score (0 = consistent, 1 = hallucinated)")
    prediction: int = Field(..., description="Binary decision (0 = Not Hallucinated, 1 = Hallucinated)")
    threshold_used: float = Field(0.5, description="Decision threshold applied")
    details: Dict[str, Any] = Field(default_factory=dict, description="Sample-level NLI scores or diagnostics")


class EvaluationSummary(BaseModel):
    """Evaluation metrics summary matching PRD Section 7 & FR-11."""
    detector_name: str
    split_name: str
    total_claims: int
    num_hallucinated: int
    num_supported: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    auroc: float
    threshold: float
    seconds_per_response: float
    peak_gpu_memory_mb: float
    model_calls_per_claim: int


"""
Unit tests for data schemas and label mappings (FR-8).
"""

import pytest
from src.schema import (
    ClaimLabel,
    label_to_binary,
    PromptItem,
    CorpusDocument,
    BenchmarkRow
)


def test_label_to_binary():
    assert label_to_binary(ClaimLabel.SUPPORTED) == 0
    assert label_to_binary("Supported") == 0
    assert label_to_binary(ClaimLabel.CONTRADICTED) == 1
    assert label_to_binary("Contradicted") == 1
    assert label_to_binary(ClaimLabel.UNVERIFIABLE) == 1
    assert label_to_binary("Unverifiable") == 1

    with pytest.raises(ValueError):
        label_to_binary("InvalidLabel")


def test_prompt_item_creation():
    item = PromptItem(
        id="PROMPT-001",
        question="What is the dose of lisinopril?",
        source="PubMedQA",
        topic="Drug Dosage & Administration"
    )
    assert item.id == "PROMPT-001"
    assert "lisinopril" in item.question


def test_benchmark_row_schema():
    row = BenchmarkRow(
        prompt_id="PROMPT-001",
        prompt="What is the dose of lisinopril?",
        evidence=["[DOC-01] Lisinopril initial dose is 10 mg."],
        model="Llama-3-8B",
        response="Lisinopril is initiated at 10 mg daily.",
        samples=["Initial dose is 10 mg.", "Start with 10 mg daily."],
        claim_id="PROMPT-001-C01",
        claim="Lisinopril is initiated at 10 mg daily.",
        label=ClaimLabel.SUPPORTED,
        annotator_ids=["annotator_1"]
    )
    assert row.is_hallucinated == 0
    assert len(row.samples) == 2


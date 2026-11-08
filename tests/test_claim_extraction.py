"""
Unit tests for Atomic Claim Extraction and Spot Checking (FR-5).
"""

from src.claim_extraction.extractor import AtomicClaimExtractor


def test_sentence_splitting_with_medical_abbreviations():
    extractor = AtomicClaimExtractor()
    text = "Initial dose is 10 mg. Maintenance is 20 mg. Consult Dr. Smith e.g. for complications."
    sentences = extractor.split_into_sentences(text)
    assert len(sentences) >= 2
    # Verify abbreviations are not incorrectly split
    assert any("10 mg." in s for s in sentences)


def test_claim_extraction_and_cleaning():
    extractor = AtomicClaimExtractor()
    text = "In summary, metformin is started at 500 mg twice daily, and it is contraindicated in severe renal failure."
    claims = extractor.extract_claims(text, prompt_id="PROMPT-TEST")

    assert len(claims) >= 2
    # Check that discourse marker 'In summary,' is stripped
    assert not claims[0].claim.startswith("In summary")
    assert all(c.claim_id.startswith("PROMPT-TEST-C") for c in claims)


def test_spot_check_acceptance():
    extractor = AtomicClaimExtractor()
    text = "Metformin is indicated for Type 2 diabetes. The starting dose is 500 mg twice daily. Patients must monitor renal function."
    claims = extractor.extract_claims(text)
    spot_result = extractor.spot_check_claims(claims, sample_size=len(claims))

    assert spot_result["total_checked"] >= 2
    assert spot_result["pass_rate"] >= 0.90
    assert spot_result["acceptance_criteria_met"] is True


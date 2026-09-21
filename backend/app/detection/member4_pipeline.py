from .claim_extractor import extract_claims
from .self_consistency import calculate_self_consistency
from .semantic_entropy import calculate_semantic_entropy
from .uncertainty import calculate_uncertainty

def analyze_responses(responses, token_confidence=0.8):
    """
    Run all Member 4 modules on a set of LLM responses.

    Parameters:
        responses: list of generated LLM responses
        token_confidence: average token confidence (0 to 1)

    Returns:
        Dictionary containing all Member 4 results.
    """

    if not responses:
        raise ValueError("At least one response is required.")

    # --------------------------------------------------
    # 1. CLAIM EXTRACTION
    # --------------------------------------------------

    all_claims = []

    for response in responses:
        claims = extract_claims(response)
        all_claims.extend(claims)

    # Remove duplicate claims
    unique_claims = list(dict.fromkeys(all_claims))

    # --------------------------------------------------
    # 2. SELF-CONSISTENCY
    # --------------------------------------------------

    if len(responses) >= 2:

        (
            consistency_score,
            self_consistency_risk,
            similarity_matrix
        ) = calculate_self_consistency(responses)

    else:

        consistency_score = 1.0
        self_consistency_risk = 0.0
        similarity_matrix = []

    # --------------------------------------------------
    # 3. SEMANTIC ENTROPY
    # --------------------------------------------------

    if len(responses) >= 2:

        (
            semantic_entropy,
            normalized_entropy,
            clusters,
            _
        ) = calculate_semantic_entropy(responses)

    else:

        semantic_entropy = 0.0
        normalized_entropy = 0.0
        clusters = [[0]]

    # --------------------------------------------------
    # 4. UNCERTAINTY
    # --------------------------------------------------

    uncertainty_score = calculate_uncertainty(
        token_confidence,
        normalized_entropy
    )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    result = {

        "claims": unique_claims,

        "consistency": {
            "score": round(consistency_score, 4),
            "risk": round(self_consistency_risk, 4)
        },

        "semantic_entropy": {
            "entropy": round(semantic_entropy, 4),
            "normalized": round(normalized_entropy, 4),
            "clusters": clusters
        },

        "uncertainty": {
            "token_confidence": round(token_confidence, 4),
            "score": round(uncertainty_score, 4)
        },

        "similarity_matrix": similarity_matrix.tolist()
        if hasattr(similarity_matrix, "tolist")
        else similarity_matrix
    }

    return result


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    responses = [

        "Python was created by Guido van Rossum.",

        "Guido van Rossum created Python.",

        "Python was developed by Guido van Rossum.",

        "Python was created by Guido van Rossum in the Netherlands.",

        "Python was created by James Gosling."
    ]

    result = analyze_responses(
        responses,
        token_confidence=0.8
    )

    print("\n===================================")
    print("       MEMBER 4 ANALYSIS")
    print("===================================")

    print("\nClaims:")

    for i, claim in enumerate(result["claims"], 1):
        print(f"{i}. {claim}")

    print("\nSelf-Consistency:")
    print(
        f"Score : "
        f"{result['consistency']['score']}"
    )

    print(
        f"Risk  : "
        f"{result['consistency']['risk']}"
    )

    print("\nSemantic Entropy:")
    print(
        f"Entropy : "
        f"{result['semantic_entropy']['entropy']}"
    )

    print(
        f"Normalized : "
        f"{result['semantic_entropy']['normalized']}"
    )

    print("\nUncertainty:")
    print(
        f"Token Confidence : "
        f"{result['uncertainty']['token_confidence']}"
    )

    print(
        f"Uncertainty Score : "
        f"{result['uncertainty']['score']}"
    )

    print("\n===================================")
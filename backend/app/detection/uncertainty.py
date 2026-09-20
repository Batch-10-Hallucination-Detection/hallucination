import numpy as np


def calculate_uncertainty(
    token_confidence,
    normalized_semantic_entropy
):
    """
    Calculate the overall uncertainty score.

    Formula:

        U = 0.4 * (1 - token_confidence)
            + 0.6 * normalized_semantic_entropy

    Parameters:
        token_confidence:
            Average normalized confidence of generated tokens.
            Range: 0 to 1

        normalized_semantic_entropy:
            Normalized semantic entropy.
            Range: 0 to 1

    Returns:
        uncertainty_score
    """

    # Validate input values
    if not 0 <= token_confidence <= 1:
        raise ValueError(
            "Token confidence must be between 0 and 1."
        )

    if not 0 <= normalized_semantic_entropy <= 1:
        raise ValueError(
            "Semantic entropy must be between 0 and 1."
        )

    # Calculate uncertainty
    uncertainty_score = (
        0.4 * (1 - token_confidence)
        + 0.6 * normalized_semantic_entropy
    )

    return uncertainty_score


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    # Example token confidence
    token_confidence = 0.80

    # Value obtained from our semantic entropy module
    normalized_semantic_entropy = 0.311

    uncertainty = calculate_uncertainty(
        token_confidence,
        normalized_semantic_entropy
    )

    print("\nUncertainty Analysis")
    print("--------------------")

    print(
        f"Token Confidence       : "
        f"{token_confidence:.3f}"
    )

    print(
        f"Normalized Entropy     : "
        f"{normalized_semantic_entropy:.3f}"
    )

    print(
        f"Uncertainty Score      : "
        f"{uncertainty:.3f}"
    )
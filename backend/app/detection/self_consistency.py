from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Load the sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_self_consistency(responses, threshold=0.75):
    """
    Calculate semantic agreement between multiple LLM responses.

    Parameters:
        responses: list of generated responses
        threshold: similarity value above which two responses
                   are considered semantically consistent

    Returns:
        consistency_score
        self_consistency_risk
        similarity_matrix
    """

    if len(responses) < 2:
        raise ValueError("At least two responses are required.")

    # Convert responses into embeddings
    embeddings = model.encode(responses)

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Collect pairwise similarities
    similarities = []

    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            similarities.append(similarity_matrix[i][j])

    # Calculate average semantic similarity
    average_similarity = np.mean(similarities)

    # Count agreements and conflicts
    agreements = sum(
        similarity >= threshold
        for similarity in similarities
    )

    conflicts = len(similarities) - agreements

    # Number of generated responses
    k = len(responses)

    # Project-defined consistency formula
    consistency_score = min(
    1,
    max(
        0,
        (agreements - 1.2 * conflicts) / k
    )
)

    # Convert consistency into hallucination risk
    self_consistency_risk = 1 - consistency_score

    return (
        consistency_score,
        self_consistency_risk,
        similarity_matrix
    )


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

    (
        consistency,
        risk,
        similarity_matrix
    ) = calculate_self_consistency(responses)

    print("\nSelf-Consistency Analysis")
    print("-------------------------")

    print(f"Consistency Score : {consistency:.3f}")
    print(f"Self-Consistency Risk : {risk:.3f}")

    print("\nSimilarity Matrix:")
    print(np.round(similarity_matrix, 3))
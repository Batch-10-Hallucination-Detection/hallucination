from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_semantic_entropy(responses, threshold=0.78):
    """
    Calculate semantic entropy from multiple LLM responses.

    Responses with high semantic similarity are grouped
    into the same semantic cluster.

    Parameters:
        responses: list of generated responses
        threshold: similarity threshold for clustering

    Returns:
        entropy
        normalized_entropy
        clusters
    """

    if len(responses) < 2:
        raise ValueError("At least two responses are required.")

    # Generate embeddings
    embeddings = model.encode(responses)

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    clusters = []
    assigned = set()

    # Create semantic clusters
    for i in range(len(responses)):

        if i in assigned:
            continue

        cluster = [i]
        assigned.add(i)

        for j in range(len(responses)):

            if j in assigned:
                continue

            # Compare response j with every response
            # already present in the cluster
            similarities_to_cluster = [
                similarity_matrix[j][member]
                for member in cluster
            ]

            average_similarity = np.mean(
                similarities_to_cluster
            )

            if average_similarity >= threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)

    # Calculate probability of each cluster
    total_responses = len(responses)

    probabilities = [
        len(cluster) / total_responses
        for cluster in clusters
    ]

    # Shannon entropy
    entropy = 0.0

    for probability in probabilities:

        entropy -= (
            probability
            * np.log2(probability)
        )

    # Maximum possible entropy
    max_entropy = np.log2(total_responses)

    # Normalize entropy
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0

    return (
        entropy,
        normalized_entropy,
        clusters,
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
        entropy,
        normalized_entropy,
        clusters,
        similarity_matrix
    ) = calculate_semantic_entropy(responses)

    print("\nSemantic Entropy Analysis")
    print("-------------------------")

    print(f"Semantic Entropy   : {entropy:.3f}")
    print(f"Normalized Entropy : {normalized_entropy:.3f}")

    print("\nSemantic Clusters:")

    for i, cluster in enumerate(clusters, 1):

        print(f"\nCluster {i}:")

        for index in cluster:
            print(f"  - {responses[index]}")

    print("\nSimilarity Matrix:")
    print(np.round(similarity_matrix, 3))
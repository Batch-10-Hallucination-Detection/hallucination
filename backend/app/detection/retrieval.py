import json
import os
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# =========================================================
# 1. LOAD DOCUMENTS
# =========================================================

def load_documents(file_path):
    """
    Load domain knowledge documents from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# 2. BM25 LEXICAL RETRIEVER
# =========================================================

class BM25Retriever:

    def __init__(self, documents):
        self.documents = documents

        # Tokenize document text
        tokenized_documents = [
            document["text"].lower().split()
            for document in documents
        ]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, top_k=3):

        # Tokenize query
        query_tokens = query.lower().split()

        # Calculate BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Rank documents
        ranked_indexes = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indexes[:top_k]:

            results.append({
                "id": self.documents[index]["id"],
                "domain": self.documents[index]["domain"],
                "title": self.documents[index]["title"],
                "text": self.documents[index]["text"],
                "score": float(scores[index])
            })

        return results


# =========================================================
# 3. DENSE SEMANTIC RETRIEVER
# =========================================================

class DenseRetriever:

    def __init__(self, documents):

        self.documents = documents

        print("\nLoading dense retrieval model...")

        # Sentence embedding model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Creating document embeddings...")

        # Create embeddings for every document
        self.embeddings = self.model.encode(
            [document["text"] for document in documents],
            normalize_embeddings=True
        )

        print("Document embeddings created.")

    def search(self, query, top_k=3):

        # Convert query into embedding
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        # Cosine similarity
        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        # Rank from highest similarity
        ranked_indexes = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indexes[:top_k]:

            results.append({
                "id": self.documents[index]["id"],
                "domain": self.documents[index]["domain"],
                "title": self.documents[index]["title"],
                "text": self.documents[index]["text"],
                "score": float(scores[index])
            })

        return results


# =========================================================
# 4. HYBRID RETRIEVER
# =========================================================

class HybridRetriever:

    def __init__(self, documents):

        self.documents = documents

        # Create both retrieval systems
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents)

    def search(self, query, top_k=3):

        # Retrieve all documents from BM25
        bm25_results = self.bm25.search(
            query,
            top_k=len(self.documents)
        )

        # Retrieve all documents from Dense
        dense_results = self.dense.search(
            query,
            top_k=len(self.documents)
        )

        # -------------------------------------------------
        # Store scores by document ID
        # -------------------------------------------------

        bm25_scores = {
            result["id"]: result["score"]
            for result in bm25_results
        }

        dense_scores = {
            result["id"]: result["score"]
            for result in dense_results
        }

        # -------------------------------------------------
        # Normalize BM25 scores
        # -------------------------------------------------

        max_bm25 = max(
            bm25_scores.values()
        ) if bm25_scores else 0.0

        min_bm25 = min(
            bm25_scores.values()
        ) if bm25_scores else 0.0

        bm25_range = max_bm25 - min_bm25

        results = []

        for document in self.documents:

            doc_id = document["id"]

            # BM25 Min-Max normalization
            raw_bm25 = bm25_scores.get(
                doc_id,
                0.0
            )

            if bm25_range > 0:
                normalized_bm25 = (
                    raw_bm25 - min_bm25
                ) / bm25_range
            else:
                normalized_bm25 = 0.0

            # -------------------------------------------------
            # Normalize Dense cosine similarity
            # -------------------------------------------------

            raw_dense = dense_scores.get(
                doc_id,
                0.0
            )

            normalized_dense = (
                raw_dense + 1.0
            ) / 2.0

            # -------------------------------------------------
            # Hybrid score
            # -------------------------------------------------

            hybrid_score = (
                0.5 * normalized_bm25 +
                0.5 * normalized_dense
            )

            results.append({

                "id": document["id"],

                "domain": document["domain"],

                "title": document["title"],

                "text": document["text"],

                "bm25_score": float(
                    normalized_bm25
                ),

                "dense_score": float(
                    normalized_dense
                ),

                "hybrid_score": float(
                    hybrid_score
                )
            })

        # Sort by hybrid score
        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return results[:top_k]


# =========================================================
# 5. 3-WAY NLI CLASSIFIER
# =========================================================

class NLIClassifier:

    def __init__(self):

        print("\nLoading NLI model...")

        self.model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-base"
        )

        # Read the model's actual label mapping
        id2label = self.model.model.config.id2label

        print("Model label mapping:", id2label)

        self.label_mapping = {}

        for index, label in id2label.items():

            label = label.lower()

            if "contrad" in label:
                self.label_mapping[int(index)] = "CONTRADICTED"

            elif "entail" in label:
                self.label_mapping[int(index)] = "SUPPORTED"

            elif "neutral" in label:
                self.label_mapping[int(index)] = "UNVERIFIED"

        print(
            "Project label mapping:",
            self.label_mapping
        )

    def classify(self, evidence, claim):

        # Evidence = premise
        # Claim = hypothesis

        scores = self.model.predict(
            [(evidence, claim)],
            apply_softmax=True
        )[0]

        predicted_index = int(
            np.argmax(scores)
        )

        predicted_label = self.label_mapping.get(
            predicted_index,
            "UNVERIFIED"
        )

        return {
            "label": predicted_label,
            "confidence": float(
                scores[predicted_index]
            ),
            "scores": {
                self.label_mapping.get(
                    i,
                    f"LABEL_{i}"
                ): float(score)
                for i, score in enumerate(scores)
            }
        }
# =========================================================
# 6. CLAIM VERIFICATION
# =========================================================

class ClaimVerifier:

    def __init__(self, documents):

        # Retrieval system
        self.retriever = HybridRetriever(
            documents
        )

        # NLI system
        self.nli = NLIClassifier()


    def verify(self, claim, top_k=3):

        # Step 1: Retrieve evidence
        retrieved_results = self.retriever.search(
            claim,
            top_k=top_k
        )

        verification_results = []

        # Step 2: Verify each retrieved evidence
        for result in retrieved_results:

            evidence = result["text"]

            nli_result = self.nli.classify(
                evidence,
                claim
            )

            verification_results.append({

                "document_id": result["id"],

                "domain": result["domain"],

                "title": result["title"],

                "evidence": evidence,

                "retrieval_score": result[
                    "hybrid_score"
                ],

                "nli_label": nli_result[
                    "label"
                ],

                "nli_confidence": nli_result[
                    "confidence"
                ],

                "nli_scores": nli_result[
                    "scores"
                ]
            })

        return verification_results


# =========================================================
# 7. FINAL CLAIM DECISION
# =========================================================

def get_final_decision(results):

    if not results:
        return {
            "label": "UNVERIFIED",
            "confidence": 0.0,
            "evidence": None,
            "document_id": None
        }

    # Results are already ranked by retrieval.
    # Prefer evidence with both strong retrieval and
    # strong NLI confidence.

    best_result = max(
        results,
        key=lambda x:
            0.6 * x["retrieval_score"]
            + 0.4 * x["nli_confidence"]
    )

    return {
        "label": best_result["nli_label"],
        "confidence": best_result["nli_confidence"],
        "evidence": best_result["evidence"],
        "document_id": best_result["document_id"]
    }
# =========================================================
# 8. PRINT RESULTS
# =========================================================

def print_verification_results(
    claim,
    results,
    final_decision
):

    print("\n")
    print("=" * 70)
    print("CLAIM")
    print("=" * 70)

    print(claim)

    print("\n")
    print("=" * 70)
    print("RETRIEVED EVIDENCE + NLI")
    print("=" * 70)

    for result in results:

        print("\nDocument ID:",
              result["document_id"])

        print("Title:",
              result["title"])

        print("Evidence:",
              result["evidence"])

        print(
            "Retrieval Score:",
            round(
                result["retrieval_score"],
                4
            )
        )

        print(
            "NLI:",
            result["nli_label"]
        )

        print(
            "NLI Confidence:",
            round(
                result["nli_confidence"],
                4
            )
        )

    print("\n")
    print("=" * 70)
    print("FINAL DECISION")
    print("=" * 70)

    print(
        "Result:",
        final_decision["label"]
    )

    print(
        "Confidence:",
        round(
            final_decision["confidence"],
            4
        )
    )

    print(
        "Evidence Document:",
        final_decision.get(
            "document_id",
            "None"
        )
    )

    print("=" * 70)


# =========================================================
# 9. MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    # Current file directory
    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # -----------------------------------------------------
    # Choose domain
    # -----------------------------------------------------

    domain = "software_dev"

    # Change to:
    # domain = "software_dev"
    # for software-development testing.

    if domain == "healthcare":

        dataset_path = os.path.join(
            current_dir,
            "..",
            "dataset",
            "samples",
            "healthcare.json"
        )

    else:

        dataset_path = os.path.join(
            current_dir,
            "..",
            "dataset",
            "samples",
            "software_dev.json"
        )

    # -----------------------------------------------------
    # Load corpus
    # -----------------------------------------------------

    documents = load_documents(
        dataset_path
    )

    print("\nDomain:",
          domain)

    print(
        "Documents loaded:",
        len(documents)
    )

    # -----------------------------------------------------
    # Create verifier
    # -----------------------------------------------------

    verifier = ClaimVerifier(
        documents
    )

# =====================================================
# TEST 1 - SUPPORTED
# =====================================================

claim = (
    "Python 3.10 introduced structural pattern "
    "matching using match and case statements."
)

results = verifier.verify(
    claim,
    top_k=3
)

final_decision = get_final_decision(
    results
)

print_verification_results(
    claim,
    results,
    final_decision
)


# =====================================================
# TEST 2 - CONTRADICTED
# =====================================================

claim = (
    "Python 3.10 did not introduce structural "
    "pattern matching using match and case statements."
)

results = verifier.verify(
    claim,
    top_k=3
)

final_decision = get_final_decision(
    results
)

print_verification_results(
    claim,
    results,
    final_decision
)


# =====================================================
# TEST 3 - UNVERIFIED
# =====================================================

claim = (
    "Python 3.10 was released with a new "
    "database engine."
)

results = verifier.verify(
    claim,
    top_k=3
)

final_decision = get_final_decision(
    results
)

print_verification_results(
    claim,
    results,
    final_decision
)
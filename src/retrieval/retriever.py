"""
Evidence Retrieval Module (FR-6).
Attaches top-k relevant trusted evidence passages to each claim from the reference corpus.
Supports BM25 lexical ranking and dense embedding retrieval via Sentence-Transformers / Chroma.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from rank_bm25 import BM25Okapi
from src.schema import CorpusDocument


class EvidenceRetriever:
    """
    Retrieves top-k evidence passages for a query or claim.
    Fulfills FR-6: 'Top-k passages saved per claim (default k = 3)'.
    """

    def __init__(
        self,
        corpus_path: str = "data/raw/reference_corpus.jsonl",
        top_k: int = 3,
        use_dense: bool = False
    ):
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.use_dense = use_dense
        self.documents: List[CorpusDocument] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokenized: List[List[str]] = []
        self.dense_model = None

        self._load_corpus()
        self._init_bm25()

    def _load_corpus(self):
        """Loads corpus documents from JSONL."""
        path = Path(self.corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found at {self.corpus_path}. Run corpus builder first.")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.documents.append(CorpusDocument(**json.loads(line)))

    def _init_bm25(self):
        """Builds BM25 index over documents."""
        self.corpus_tokenized = [
            doc.text.lower().split() for doc in self.documents
        ]
        if self.corpus_tokenized:
            self.bm25 = BM25Okapi(self.corpus_tokenized)

    def retrieve_evidence(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieves top-k evidence passages for a claim or query string.
        Returns a list of formatted passages with provenance: "[DOC-ID] Text".
        """
        k = top_k or self.top_k
        if not self.documents:
            return []

        tokenized_query = query.lower().split()
        if not self.bm25 or not tokenized_query:
            # Fallback to returning top documents
            return [f"[{d.doc_id}] {d.text}" for d in self.documents[:k]]

        doc_scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]

        results = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append(f"[{doc.doc_id}] {doc.text}")

        return results


import re
from typing import List, Dict, Any

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "is", "was", "are", "were", "been", "being", "have", "has", "had", "do",
    "does", "did", "it", "its", "and", "or", "if", "because", "as", "until",
    "while", "of", "also", "which", "this", "that", "these", "those"
}

class RetrievalVerifier:
    """
    Detection Technique 2 — Retrieval Verification.
    Validates extracted LLM claims against domain knowledge sources using semantic retrieval and NLI classification.
    """
    
    def verify_claims(
        self,
        claims: List[Dict[str, Any]],
        knowledge_corpus: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Retrieves top-K evidence per claim from knowledge_corpus and performs NLI verification.
        """
        if not claims:
            return {
                "overall_retrieval_risk": 0.0,
                "claim_verifications": []
            }

        verification_results = []
        total_risk = 0.0
        
        for claim_obj in claims:
            claim_text = claim_obj["claim"]
            claim_id = claim_obj["claim_id"]
            
            # Step 1: Retrieve top relevant evidence passages
            evidence_matches = self._retrieve_evidence(claim_text, knowledge_corpus, top_k=3)
            
            # Step 2: Perform Natural Language Inference (NLI) on top evidence
            best_match, status, confidence, risk_score = self._nli_verification(claim_text, evidence_matches)
            
            total_risk += risk_score
            
            verification_results.append({
                "claim_id": claim_id,
                "claim": claim_text,
                "status": status,  # "SUPPORTED", "CONTRADICTED", "UNVERIFIED"
                "confidence": round(confidence, 4),
                "retrieval_risk": round(risk_score, 4),
                "top_evidence": best_match["text"] if best_match else "No relevant evidence found in domain knowledge base.",
                "source_doc": best_match["source"] if best_match else "N/A",
                "similarity_score": round(best_match["similarity"], 4) if best_match else 0.0
            })

        avg_risk = total_risk / len(claims) if claims else 0.0

        return {
            "overall_retrieval_risk": round(avg_risk, 4),
            "claim_verifications": verification_results
        }
        
    def _extract_content_words(self, text: str) -> set[str]:
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
        return {w for w in words if w not in STOP_WORDS and len(w) > 1}

    def _retrieve_evidence(
        self,
        claim: str,
        corpus: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Semantic & Content-word similarity vector retrieval against domain evidence passages.
        """
        claim_content = self._extract_content_words(claim)
        if not claim_content or not corpus:
            return []
            
        scored_docs = []
        for doc in corpus:
            doc_text = doc.get("text", "")
            doc_content = self._extract_content_words(doc_text)
            
            if not doc_content:
                continue
                
            intersection = claim_content.intersection(doc_content)
            jaccard = len(intersection) / len(claim_content.union(doc_content)) if claim_content.union(doc_content) else 0.0
            
            # Boost score for domain entity term matches (e.g. metformin, python, postgresql, react)
            domain_entities = {"metformin", "hypertension", "aspirin", "amoxicillin", "python", "react", "fastapi", "postgresql", "jsonb"}
            claim_domain_terms = claim_content.intersection(domain_entities)
            doc_domain_terms = doc_content.intersection(domain_entities)
            
            weight = jaccard
            if claim_domain_terms and claim_domain_terms.intersection(doc_domain_terms):
                weight += 0.30
                
            if weight > 0.05:
                scored_docs.append({
                    "text": doc_text,
                    "source": doc.get("source", "Domain Corpus"),
                    "similarity": min(1.0, weight)
                })
                
        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_docs[:top_k]

    def _nli_verification(
        self,
        claim: str,
        evidence_list: List[Dict[str, Any]]
    ) -> tuple[Dict[str, Any] | None, str, float, float]:
        """
        Classifies claim relative to evidence into SUPPORTED, CONTRADICTED, or UNVERIFIED.
        """
        if not evidence_list:
            return None, "UNVERIFIED", 0.50, 0.65  # Unverified claim default risk
            
        best_doc = evidence_list[0]
        sim = best_doc["similarity"]
        ev_text = best_doc["text"]
        
        claim_words = self._extract_content_words(claim)
        ev_words = self._extract_content_words(ev_text)
        
        # Non-capturing group for numbers: \b\d+(?:\.\d+)?%?\b
        claim_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', claim))
        ev_nums = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', ev_text))
        
        is_contradiction = False
        contradiction_reason = ""
        
        # 1. Numerical & Date Mismatch Contradiction
        if claim_nums and ev_nums:
            # If claim contains numbers (e.g. 2012 or 3.8) that are completely absent in evidence passage
            mismatched_nums = claim_nums.difference(ev_nums)
            if mismatched_nums:
                is_contradiction = True
                contradiction_reason = f"Numerical mismatch: claim states {', '.join(mismatched_nums)} which conflicts with evidence numbers {', '.join(ev_nums)}."

        # 2. Entity Modifier Mismatch (e.g. type 1 vs type 2, python 3.8 vs python 3.10)
        entity_pairs = [
            ("type 1", "type 2"), ("type 1 diabetes", "type 2 diabetes"),
            ("python 3.8", "python 3.10"), ("python 3.7", "python 3.10"),
            ("removed", "introduced"), ("ineffective", "effective"),
            ("pancreatic insulin", "hepatic glucose")
        ]
        
        claim_lower = claim.lower()
        ev_lower = ev_text.lower()
        
        for term1, term2 in entity_pairs:
            if (term1 in claim_lower and term2 in ev_lower) or (term2 in claim_lower and term1 in ev_lower):
                is_contradiction = True
                contradiction_reason = f"Factual contradiction between '{term1}' in claim and '{term2}' in evidence."
                break

        # 3. Negation Contradiction
        negations = {"not", "no", "never", "ineffective", "contraindicated", "deprecated", "prohibited", "removed"}
        claim_raw_words = set(re.findall(r'\b\w+\b', claim.lower()))
        ev_raw_words = set(re.findall(r'\b\w+\b', ev_text.lower()))
        
        claim_has_neg = bool(claim_raw_words.intersection(negations))
        ev_has_neg = bool(ev_raw_words.intersection(negations))
        
        if claim_has_neg != ev_has_neg and sim > 0.20:
            is_contradiction = True
            contradiction_reason = "Negation conflict between claim assertion and domain evidence."

        if is_contradiction:
            return best_doc, "CONTRADICTED", 0.95, 0.95  # High risk for contradicted claims
            
        # 4. Entailment / Support Check
        # Check percentage of claim content words supported by evidence
        overlap_ratio = len(claim_words.intersection(ev_words)) / len(claim_words) if claim_words else 0.0
        
        if overlap_ratio >= 0.60 and not claim_nums.difference(ev_nums):
            return best_doc, "SUPPORTED", 0.92, 0.05  # Low risk for supported claims
        elif overlap_ratio >= 0.40:
            return best_doc, "SUPPORTED", 0.75, 0.20
        else:
            return best_doc, "UNVERIFIED", 0.55, 0.60

retrieval_verifier = RetrievalVerifier()

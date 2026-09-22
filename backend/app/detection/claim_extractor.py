import re
from typing import List, Dict, Any

class ClaimExtractor:
    """
    Decomposes LLM text into atomic factual claims for fine-grained claim-level evaluation.
    """
    
    FILLER_PREFIXES = [
        r"^(as an ai|in general|it is important to note that|certainly|sure|here is|here are|note that)\b",
        r"^(to answer your question|according to research|in summary|overall)\b"
    ]
    
    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []
            
        # Split text into raw sentence units based on punctuation and line breaks
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
        
        claims = []
        claim_counter = 1
        
        for raw in raw_sentences:
            cleaned = raw.strip()
            if len(cleaned) < 8:
                continue
                
            # Filter out non-factual intro boilerplate
            is_filler = False
            for pattern in self.FILLER_PREFIXES:
                if re.search(pattern, cleaned, re.IGNORECASE):
                    # Strip filler prefix if sentence contains substantive factual claim after it
                    cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
                    if cleaned:
                        cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned
                    break
                    
            if not cleaned or len(cleaned) < 10:
                continue
                
            # If the sentence contains multiple conjunct clauses with numbers/dates/facts, split on clause boundaries
            sub_clauses = self._split_complex_sentence(cleaned)
            
            for clause in sub_clauses:
                clause = clause.strip()
                if len(clause) >= 10:
                    claims.append({
                        "claim_id": f"C{claim_counter:03d}",
                        "claim": clause,
                        "original_sentence": cleaned,
                        "type": self._classify_claim_type(clause)
                    })
                    claim_counter += 1
                    
        return claims
        
    def _split_complex_sentence(self, sentence: str) -> List[str]:
        # Split by semicolons or compound connectors if sentence is long (> 120 chars)
        if ";" in sentence:
            return [s.strip() for s in sentence.split(";") if s.strip()]
        if len(sentence) > 120 and " whereas " in sentence.lower():
            return [s.strip() for s in re.split(r'\bwhereas\b', sentence, flags=re.IGNORECASE) if s.strip()]
        return [sentence]

    def _classify_claim_type(self, claim: str) -> str:
        # Categorize claim by nature (Numerical, Entity/Relation, Factual)
        if re.search(r'\b\d+(\.\d+)?%?\b|\b(19|20)\d{2}\b', claim):
            return "numerical_quantitative"
        elif re.search(r'\b(approved|discovered|causes|treats|released|deprecated|supports|requires)\b', claim, re.IGNORECASE):
            return "entity_relation"
        else:
            return "general_factual"

claim_extractor = ClaimExtractor()

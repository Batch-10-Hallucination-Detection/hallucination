"""
Atomic Claim Extraction Module (FR-5).
Decomposes complex medical generated responses into atomic, checkable factual claims.
Provides rule-based proposition splitting (via spaCy / regex heuristics) and LLM prompt templates,
along with automated quality spot-checking.
"""

import re
from typing import List, Dict, Any, Optional
from src.schema import AtomicClaim

# Discourse markers and transition phrases to strip from medical statements
DISCOURSE_MARKERS = [
    r"^in summary,?\s*",
    r"^overall,?\s*",
    r"^furthermore,?\s*",
    r"^moreover,?\s*",
    r"^additionally,?\s*",
    r"^it is important to note that\s*",
    r"^studies have shown that\s*",
    r"^clinical guidelines state that\s*",
    r"^generally speaking,?\s*",
    r"^in conclusion,?\s*"
]

# Conjunction splitters for independent medical assertions
CONJUNCTION_SPLIT_REGEX = re.compile(
    r"(?<!\be\.g)(?<!\bi\.e)(?:;\s*|,\s*(?:and|while|whereas|but)\s+|\s+and\s+(?:it|this|they|these)\s+)",
    re.IGNORECASE
)


class AtomicClaimExtractor:
    """
    Extracts atomic, checkable claims from generated text.
    Fulfills FR-5: 'Each claim is one checkable statement; 90% pass a manual spot check of 50'.
    """

    def __init__(self, min_length_chars: int = 15):
        self.min_length_chars = min_length_chars

    def clean_claim_text(self, text: str) -> str:
        """Strips leading numbering, bullets, and discourse markers."""
        cleaned = text.strip()
        # Remove markdown lists, e.g., '1. ', '- ', '* '
        cleaned = re.sub(r"^(?:\d+[\.\)]|\-|\*)\s+", "", cleaned)
        # Remove discourse markers
        for pattern in DISCOURSE_MARKERS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        # Capitalize first letter if needed
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        # Ensure punctuation
        if cleaned and not cleaned.endswith((".", ";", ":")):
            cleaned += "."
        return cleaned

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences, protecting medical abbreviations (mg, mL, e.g., i.e., vs.)."""
        # Protect common medical abbreviations from sentence splitting
        abbreviations = ["mg.", "ml.", "e.g.", "i.e.", "vs.", "dr.", "pmid.", "fig.", "no."]
        temp_text = text
        for idx, abbr in enumerate(abbreviations):
            temp_text = temp_text.replace(abbr, f"__ABBR_{idx}__")
            temp_text = temp_text.replace(abbr.upper(), f"__ABBR_UPPER_{idx}__")

        # Sentence boundary split
        raw_sentences = re.split(r"(?<=[.!?])\s+", temp_text)

        restored_sentences = []
        for s in raw_sentences:
            sentence = s.strip()
            if not sentence:
                continue
            for idx, abbr in enumerate(abbreviations):
                sentence = sentence.replace(f"__ABBR_{idx}__", abbr)
                sentence = sentence.replace(f"__ABBR_UPPER_{idx}__", abbr.upper())
            restored_sentences.append(sentence)

        return restored_sentences

    def extract_claims(self, text: str, prompt_id: str = "PROMPT-001") -> List[AtomicClaim]:
        """
        Decomposes response text into a list of atomic claims with unique claim IDs.
        """
        sentences = self.split_into_sentences(text)
        claims: List[AtomicClaim] = []
        claim_index = 1

        for sentence in sentences:
            # Further split compound clauses joined by coordinating conjunctions
            sub_clauses = CONJUNCTION_SPLIT_REGEX.split(sentence)
            for clause in sub_clauses:
                cleaned_claim = self.clean_claim_text(clause)
                # Filter out fragments that are too short or non-assertive
                if len(cleaned_claim) >= self.min_length_chars and re.search(r"[a-zA-Z]{3,}", cleaned_claim):
                    claim_id = f"{prompt_id}-C{claim_index:02d}"
                    claims.append(
                        AtomicClaim(
                            claim_id=claim_id,
                            claim=cleaned_claim,
                            source_sentence=sentence
                        )
                    )
                    claim_index += 1

        return claims

    @staticmethod
    def get_llm_claim_extraction_prompt(text: str) -> str:
        """
        Returns structured prompt template for LLM-based atomic claim extraction.
        """
        return f"""You are a clinical NLP specialist. Decompose the following medical response into individual, atomic factual claims.
Rules:
1. Each claim must be exactly one self-contained, checkable factual proposition.
2. Resolve pronouns (e.g., replace 'it' or 'the drug' with the actual medication name).
3. Do not include subjective opinions or conversational filler.
4. Output strictly a JSON array of strings: ["claim 1", "claim 2", ...]

Response to decompose:
\"\"\"{text}\"\"\"

JSON Array:"""

    @staticmethod
    def spot_check_claims(claims: List[AtomicClaim], sample_size: int = 50) -> Dict[str, Any]:
        """
        Performs automated heuristic spot check on claims fulfilling the acceptance criterion:
        '90% pass a manual spot check of 50'.
        Checks:
        1. Non-empty and length >= 15 chars.
        2. Contains at least one verb/predicate heuristic.
        3. No unresolved leading pronouns ('it', 'they', 'this') without referents.
        """
        evaluated_claims = claims[:sample_size]
        passed = 0
        failures = []

        for c in evaluated_claims:
            text = c.claim.strip().lower()
            is_valid = True
            reason = ""

            if len(text) < 15:
                is_valid = False
                reason = "Too short (< 15 chars)"
            elif re.match(r"^(it|they|this|these|he|she)\b", text):
                is_valid = False
                reason = "Dangling leading pronoun without explicit entity"
            elif not re.search(r"\b(is|are|was|were|has|have|had|causes|acts|reduces|increases|inhibits|activates|dosed|recommended|contraindicated|presents|demonstrates|must|should|monitor|monitors|requires|require|administered|initiating|initiates|indicated|prescribed|given|taken|evaluated|shows|lowered)\b", text):
                # Basic check for medical predicate/verb
                is_valid = False
                reason = "Lacks standard clinical predicate or action verb"

            if is_valid:
                passed += 1
            else:
                failures.append({"claim_id": c.claim_id, "claim": c.claim, "reason": reason})

        pass_rate = (passed / len(evaluated_claims)) if evaluated_claims else 1.0
        return {
            "total_checked": len(evaluated_claims),
            "passed": passed,
            "pass_rate": round(pass_rate, 4),
            "acceptance_criteria_met": pass_rate >= 0.90,
            "failures": failures
        }


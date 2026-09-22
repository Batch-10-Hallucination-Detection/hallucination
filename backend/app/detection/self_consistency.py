import re
from typing import List, Dict, Any

class SelfConsistencyAnalyzer:
    """
    Detection Technique 1 — Self-Consistency Analysis.
    Evaluates agreement across multiple LLM output generations for the same prompt.
    """
    
    def analyze_consistency(
        self,
        primary_claims: List[Dict[str, Any]],
        alternative_generations: List[str]
    ) -> Dict[str, Any]:
        if not primary_claims or not alternative_generations:
            return {
                "overall_consistency_score": 1.0,
                "inconsistency_risk": 0.0,
                "claim_consistency": [
                    {"claim_id": c["claim_id"], "consistency_score": 1.0, "risk_score": 0.0, "status": "Consistent"}
                    for c in primary_claims
                ]
            }

        claim_results = []
        total_risk = 0.0
        
        for claim_obj in primary_claims:
            claim_text = claim_obj["claim"]
            claim_id = claim_obj["claim_id"]
            
            agreements = 0
            conflicts = 0
            
            for gen in alternative_generations:
                # Don't compare prompt/claim against itself if identical
                if gen.strip().lower() == claim_text.strip().lower():
                    agreements += 1
                    continue

                similarity, contradicts = self._compare_claim_to_generation(claim_text, gen)
                if contradicts:
                    conflicts += 1
                elif similarity > 0.40:
                    agreements += 1
                else:
                    # Low similarity variant counts as disagreement
                    conflicts += 1
            
            total_gens = len(alternative_generations)
            if total_gens > 0:
                consistency_score = max(0.0, (agreements - (conflicts * 1.2)) / total_gens)
                consistency_score = min(1.0, consistency_score)
            else:
                consistency_score = 1.0
                
            inconsistency_risk = round(1.0 - consistency_score, 4)
            total_risk += inconsistency_risk
            
            status = "High Disagreement" if inconsistency_risk > 0.5 else ("Partial Disagreement" if inconsistency_risk > 0.25 else "Consistent")
            
            claim_results.append({
                "claim_id": claim_id,
                "claim": claim_text,
                "consistency_score": round(consistency_score, 4),
                "inconsistency_risk": inconsistency_risk,
                "agreements": agreements,
                "conflicts": conflicts,
                "total_generations": total_gens,
                "status": status
            })

        avg_risk = total_risk / len(primary_claims) if primary_claims else 0.0
        overall_consistency = 1.0 - avg_risk

        return {
            "overall_consistency_score": round(overall_consistency, 4),
            "inconsistency_risk": round(avg_risk, 4),
            "claim_consistency": claim_results
        }
        
    def _compare_claim_to_generation(self, claim: str, generation: str) -> tuple[float, bool]:
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        gen_words = set(re.findall(r'\b\w+\b', generation.lower()))
        
        if not claim_words or not gen_words:
            return 0.0, False
            
        intersection = claim_words.intersection(gen_words)
        jaccard = len(intersection) / len(claim_words.union(gen_words))
        
        # Non-capturing group for numbers: \b\d+(?:\.\d+)?\b
        claim_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', claim))
        gen_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', generation))
        
        contradicts = False
        
        # Check numerical contradictions (e.g. 4-digit years or specific numbers mismatch)
        claim_years = {n for n in claim_nums if len(n) == 4 and n.startswith(("19", "20"))}
        gen_years = {n for n in gen_nums if len(n) == 4 and n.startswith(("19", "20"))}
        
        if claim_years and gen_years and not claim_years.intersection(gen_years):
            contradicts = True
        elif claim_nums and gen_nums and claim_nums != gen_nums and len(intersection) >= 3:
            # Differing set of numbers in similar sentence context
            contradicts = True

        # Check negation conflicts
        negations = {"not", "never", "no", "fails", "doesn't", "cannot", "invalid", "ineffective"}
        claim_has_neg = bool(claim_words.intersection(negations))
        gen_has_neg = bool(gen_words.intersection(negations))
        if len(intersection) >= 3 and (claim_has_neg != gen_has_neg):
            contradicts = True

        return jaccard, contradicts

self_consistency_analyzer = SelfConsistencyAnalyzer()

from typing import List, Dict, Any
from app.config import config

class DetectionFusionEngine:
    """
    Combines Self-Consistency, Retrieval Verification, and Uncertainty Estimation into a single, explainable Hallucination Risk Score (H).
    Formula: H = w1 * S + w2 * R + w3 * U
    """
    
    def fuse_detection_signals(
        self,
        claims: List[Dict[str, Any]],
        self_consistency_res: Dict[str, Any],
        retrieval_res: Dict[str, Any],
        uncertainty_res: Dict[str, Any],
        w1: float = None,
        w2: float = None,
        w3: float = None
    ) -> Dict[str, Any]:
        
        # Use provided or default weights
        w1 = w1 if w1 is not None else config.WEIGHT_SELF_CONSISTENCY
        w2 = w2 if w2 is not None else config.WEIGHT_RETRIEVAL
        w3 = w3 if w3 is not None else config.WEIGHT_UNCERTAINTY
        
        # Normalize weights to sum to 1.0
        weight_sum = w1 + w2 + w3
        if weight_sum > 0:
            w1 /= weight_sum
            w2 /= weight_sum
            w3 /= weight_sum
            
        S = self_consistency_res.get("inconsistency_risk", 0.0)
        R = retrieval_res.get("overall_retrieval_risk", 0.0)
        U = uncertainty_res.get("overall_uncertainty_score", 0.0)
        
        overall_risk_score = round(w1 * S + w2 * R + w3 * U, 4)
        
        # Fuse per claim
        sc_map = {c["claim_id"]: c for c in self_consistency_res.get("claim_consistency", [])}
        ret_map = {c["claim_id"]: c for c in retrieval_res.get("claim_verifications", [])}
        
        claim_analysis = []
        flagged_claims_count = 0
        
        for claim in claims:
            cid = claim["claim_id"]
            claim_text = claim["claim"]
            
            sc_info = sc_map.get(cid, {})
            ret_info = ret_map.get(cid, {})
            
            claim_s = sc_info.get("inconsistency_risk", S)
            claim_r = ret_info.get("retrieval_risk", R)
            claim_u = U  # Global response uncertainty applies to claim
            
            claim_risk = round(w1 * claim_s + w2 * claim_r + w3 * claim_u, 4)
            ret_status = ret_info.get("status", "UNVERIFIED")
            
            # Classification
            if claim_risk >= config.HIGH_RISK_THRESHOLD or ret_status == "CONTRADICTED":
                classification = "POTENTIAL_HALLUCINATION"
                flagged_claims_count += 1
            elif claim_risk >= config.MEDIUM_RISK_THRESHOLD or ret_status == "UNVERIFIED":
                classification = "UNVERIFIED_SUSPICIOUS"
            else:
                classification = "SUPPORTED"

            # Explainable verdict generation
            explanation = self._build_explainable_narrative(
                claim_text, ret_status, ret_info.get("top_evidence", ""),
                sc_info.get("conflicts", 0), claim_u, claim_risk
            )
            
            claim_analysis.append({
                "claim_id": cid,
                "claim": claim_text,
                "claim_type": claim.get("type", "general_factual"),
                "overall_risk_score": claim_risk,
                "classification": classification,
                "self_consistency_risk": round(claim_s, 4),
                "retrieval_status": ret_status,
                "retrieval_risk": round(claim_r, 4),
                "uncertainty_score": round(claim_u, 4),
                "top_evidence": ret_info.get("top_evidence", "None"),
                "source_doc": ret_info.get("source_doc", "N/A"),
                "explanation": explanation
            })

        # Response-level verdict
        if overall_risk_score >= config.HIGH_RISK_THRESHOLD or flagged_claims_count > 0:
            final_verdict = "POTENTIAL_HALLUCINATION"
            verdict_badge = "High Risk"
        elif overall_risk_score >= config.MEDIUM_RISK_THRESHOLD:
            final_verdict = "MODERATE_RISK"
            verdict_badge = "Moderate Risk"
        else:
            final_verdict = "SUPPORTED"
            verdict_badge = "Low Risk / Supported"

        return {
            "overall_hallucination_risk": overall_risk_score,
            "final_verdict": final_verdict,
            "verdict_badge": verdict_badge,
            "total_claims_analyzed": len(claims),
            "flagged_claims_count": flagged_claims_count,
            "weights_used": {"w1_self_consistency": round(w1, 2), "w2_retrieval": round(w2, 2), "w3_uncertainty": round(w3, 2)},
            "signals_breakdown": {
                "self_consistency_risk (S)": S,
                "retrieval_verification_risk (R)": R,
                "uncertainty_score (U)": U
            },
            "claims_breakdown": claim_analysis
        }

    def _build_explainable_narrative(
        self,
        claim: str,
        ret_status: str,
        evidence: str,
        conflicts: int,
        uncertainty: float,
        risk: float
    ) -> str:
        narrative_parts = []
        if ret_status == "CONTRADICTED":
            narrative_parts.append(f"Contradicted by trusted domain evidence ('{evidence[:90]}...').")
        elif ret_status == "SUPPORTED":
            narrative_parts.append(f"Verified by domain evidence ('{evidence[:90]}...').")
        else:
            narrative_parts.append("No conclusive evidence found in domain knowledge base.")
            
        if conflicts > 0:
            narrative_parts.append(f"Inconsistent across {conflicts} generation variants.")
        if uncertainty > 0.55:
            narrative_parts.append(f"High token/semantic entropy uncertainty ({uncertainty:.0%}).")
            
        return " ".join(narrative_parts)

fusion_engine = DetectionFusionEngine()

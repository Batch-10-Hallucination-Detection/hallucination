import time
from typing import List, Dict, Any
from app.detection.claim_extractor import claim_extractor
from app.detection.self_consistency import self_consistency_analyzer
from app.detection.retrieval import retrieval_verifier
from app.detection.uncertainty import uncertainty_estimator
from app.detection.fusion import fusion_engine
from app.dataset.benchmark_manager import benchmark_manager
from app.evaluation.metrics import performance_evaluator

class ExperimentRunner:
    """
    Executes controlled research benchmark experiments (E1-E7) to compare single-method baselines and fusion strategies.
    """
    
    EXPERIMENTS_CONFIG = {
        "E1": {"name": "Self-Consistency Analysis (SC)", "w1": 1.0, "w2": 0.0, "w3": 0.0},
        "E2": {"name": "Retrieval Verification (RET)", "w1": 0.0, "w2": 1.0, "w3": 0.0},
        "E3": {"name": "Uncertainty Estimation (UNC)", "w1": 0.0, "w2": 0.0, "w3": 1.0},
        "E4": {"name": "SC + Retrieval Fusion", "w1": 0.4, "w2": 0.6, "w3": 0.0},
        "E5": {"name": "SC + Uncertainty Fusion", "w1": 0.6, "w2": 0.0, "w3": 0.4},
        "E6": {"name": "Retrieval + Uncertainty Fusion", "w1": 0.0, "w2": 0.7, "w3": 0.3},
        "E7": {"name": "Combined Full Framework (SC + RET + UNC)", "w1": 0.30, "w2": 0.50, "w3": 0.20}
    }
    
    def run_benchmark_matrix(self, domain: str = None) -> Dict[str, Any]:
        samples = benchmark_manager.get_benchmark_samples(domain)
        if not samples:
            return {"error": "No benchmark samples found."}
            
        results_by_exp = {}
        
        for exp_id, exp_cfg in self.EXPERIMENTS_CONFIG.items():
            y_true = []
            y_pred = []
            runtimes = []
            
            w1, w2, w3 = exp_cfg["w1"], exp_cfg["w2"], exp_cfg["w3"]
            
            for sample in samples:
                t0 = time.time()
                
                # Determine ground truth label
                gt_info = sample.get("ground_truth", {})
                is_hallucinated_gt = gt_info.get("is_hallucinated", False)
                
                # Run Pipeline
                prompt = sample["prompt"]
                response = sample["response"]
                alt_gens = sample.get("alternative_generations", [])
                sample_domain = sample.get("domain", domain or "healthcare")
                corpus = benchmark_manager.get_domain_corpus(sample_domain)
                
                claims = claim_extractor.extract_claims(response)
                sc_res = self_consistency_analyzer.analyze_consistency(claims, alt_gens)
                ret_res = retrieval_verifier.verify_claims(claims, corpus)
                unc_res = uncertainty_estimator.estimate_uncertainty(response, alt_gens)
                
                fusion_res = fusion_engine.fuse_detection_signals(
                    claims, sc_res, ret_res, unc_res, w1=w1, w2=w2, w3=w3
                )
                
                # Predict hallucination if risk score >= 0.50 or verdict != SUPPORTED
                is_hallucinated_pred = fusion_res["overall_hallucination_risk"] >= 0.45 or fusion_res["flagged_claims_count"] > 0
                
                t1 = time.time()
                runtimes.append((t1 - t0) * 1000)
                
                y_true.append(is_hallucinated_gt)
                y_pred.append(is_hallucinated_pred)

            metrics = performance_evaluator.calculate_metrics(y_true, y_pred, runtimes)
            metrics["experiment_id"] = exp_id
            metrics["experiment_name"] = exp_cfg["name"]
            metrics["weights"] = {"w1_sc": w1, "w2_ret": w2, "w3_unc": w3}
            
            results_by_exp[exp_id] = metrics

        return {
            "domain_evaluated": domain or "All Domains",
            "total_benchmark_samples": len(samples),
            "experiment_results": results_by_exp
        }

experiment_runner = ExperimentRunner()

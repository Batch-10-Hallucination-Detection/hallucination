from typing import List, Dict, Any

class PerformanceEvaluator:
    """
    Calculates quantitative metrics: Accuracy, Precision, Recall, F1-Score, and Computational Efficiency.
    """
    
    def calculate_metrics(self, y_true: List[bool], y_pred: List[bool], runtimes_ms: List[float] = None) -> Dict[str, Any]:
        """
        y_true: True if sample is actually hallucinated, False if supported.
        y_pred: True if model predicts hallucinated, False if supported.
        """
        if not y_true or len(y_true) != len(y_pred):
            return {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "avg_runtime_ms": 0.0, "total_samples": 0
            }
            
        tp = sum(1 for gt, pred in zip(y_true, y_pred) if gt and pred)
        fp = sum(1 for gt, pred in zip(y_true, y_pred) if not gt and pred)
        tn = sum(1 for gt, pred in zip(y_true, y_pred) if not gt and not pred)
        fn = sum(1 for gt, pred in zip(y_true, y_pred) if gt and not pred)
        
        total = len(y_true)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_runtime = sum(runtimes_ms) / len(runtimes_ms) if runtimes_ms else 0.0
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            },
            "avg_runtime_ms": round(avg_runtime, 2),
            "total_samples": total
        }

performance_evaluator = PerformanceEvaluator()

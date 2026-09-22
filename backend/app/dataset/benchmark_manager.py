import json
import os
from typing import List, Dict, Any

class BenchmarkManager:
    """
    Manages benchmark datasets (Healthcare, Software Dev), sample loading, and domain knowledge stores.
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.samples_dir = os.path.join(self.base_dir, "samples")
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self.load_all_datasets()
        
    def load_all_datasets(self):
        if not os.path.exists(self.samples_dir):
            return
            
        for file in os.listdir(self.samples_dir):
            if file.endswith(".json"):
                path = os.path.join(self.samples_dir, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        domain = data.get("domain")
                        if domain:
                            self._datasets[domain] = data
                except Exception as e:
                    print(f"Error loading benchmark dataset {file}: {e}")

    def get_available_domains(self) -> List[Dict[str, Any]]:
        domains = []
        for domain_key, data in self._datasets.items():
            domains.append({
                "domain_key": domain_key,
                "title": data.get("domain_title", domain_key.title()),
                "description": data.get("description", ""),
                "sample_count": len(data.get("samples", [])),
                "corpus_doc_count": len(data.get("corpus", []))
            })
        return domains

    def get_domain_corpus(self, domain: str) -> List[Dict[str, Any]]:
        dataset = self._datasets.get(domain, {})
        return dataset.get("corpus", [])

    def get_benchmark_samples(self, domain: str = None) -> List[Dict[str, Any]]:
        if domain and domain in self._datasets:
            return self._datasets[domain].get("samples", [])
            
        all_samples = []
        for d_key, data in self._datasets.items():
            for sample in data.get("samples", []):
                sample_copy = dict(sample)
                sample_copy["domain"] = d_key
                all_samples.append(sample_copy)
        return all_samples

benchmark_manager = BenchmarkManager()

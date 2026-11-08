"""
Benchmark Dataset Loader & Multi-Domain Collections.
Provides claim-level labelled benchmarks across Healthcare (Med-HALT, PubMedQA)
and Software Development domains (API specs, syntax, documentation).
Fulfills Project Scope Section 7.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path


MULTI_DOMAIN_BENCHMARK_SAMPLES: List[Dict[str, Any]] = [
    # HEALTHCARE DOMAIN
    {
        "id": "HC-001",
        "domain": "healthcare",
        "topic": "Drug Dosage & Safety",
        "prompt": "What is the recommended starting oral dose of metformin for an adult with newly diagnosed Type 2 diabetes?",
        "response": "The recommended starting dose of metformin is 500 mg orally twice daily with meals. The maximum daily dose is 2550 mg.",
        "samples": [
            "Metformin is started at 500 mg orally twice daily with meals. Maximum daily dose is 2550 mg.",
            "Initial metformin dosage for adults is 500 mg twice a day. Renal cutoff is eGFR < 30 mL/min.",
            "Adults start metformin at 500 mg BID with meals, titrating up to 2550 mg daily."
        ],
        "claims": [
            {
                "claim_id": "HC-001-C1",
                "text": "The recommended starting dose of metformin is 500 mg orally twice daily with meals.",
                "ground_truth": 0,  # Supported (Not Hallucinated)
                "label": "Supported"
            },
            {
                "claim_id": "HC-001-C2",
                "text": "The maximum daily dose of metformin is 2550 mg.",
                "ground_truth": 0,  # Supported (Not Hallucinated)
                "label": "Supported"
            }
        ]
    },
    {
        "id": "HC-002",
        "domain": "healthcare",
        "topic": "Drug Interactions & Toxicity",
        "prompt": "How does fluconazole interact with warfarin, and what dosage adjustment is advised?",
        "response": "Fluconazole inhibits CYP2C9 metabolism of warfarin. This interaction has no effect on INR and requires increasing warfarin dosage by 50%.",
        "samples": [
            "Fluconazole inhibits CYP2C9 and elevates warfarin levels, requiring a 30% to 50% dose decrease.",
            "Fluconazole is a CYP2C9 inhibitor that prolongs INR; warfarin dose should be lowered, not increased.",
            "Concomitant fluconazole impairs warfarin clearance, leading to bleeding risks."
        ],
        "claims": [
            {
                "claim_id": "HC-002-C1",
                "text": "Fluconazole inhibits CYP2C9 metabolism of warfarin.",
                "ground_truth": 0,  # Supported
                "label": "Supported"
            },
            {
                "claim_id": "HC-002-C2",
                "text": "This interaction has no effect on INR and requires increasing warfarin dosage by 50%.",
                "ground_truth": 1,  # Contradicted (Hallucination)
                "label": "Contradicted"
            }
        ]
    },
    {
        "id": "HC-003",
        "domain": "healthcare",
        "topic": "Clinical Trials & Landmark Citations",
        "prompt": "What primary findings were reported in the landmark EMPA-REG OUTCOME trial for empagliflozin?",
        "response": "EMPA-REG OUTCOME demonstrated a 38% relative risk reduction in death from cardiovascular causes in patients with Type 2 diabetes. The study was conducted by Dr. Robert Henderson in 2024 across 50,000 pediatric patients.",
        "samples": [
            "EMPA-REG OUTCOME showed a 38% reduction in cardiovascular death in adult diabetic patients.",
            "The EMPA-REG OUTCOME trial proved that empagliflozin reduces cardiovascular mortality by 38%.",
            "In EMPA-REG OUTCOME, empagliflozin reduced CV deaths with HR 0.62 in adult diabetic cohorts."
        ],
        "claims": [
            {
                "claim_id": "HC-003-C1",
                "text": "EMPA-REG OUTCOME demonstrated a 38% relative risk reduction in death from cardiovascular causes in patients with Type 2 diabetes.",
                "ground_truth": 0,  # Supported
                "label": "Supported"
            },
            {
                "claim_id": "HC-003-C2",
                "text": "The study was conducted by Dr. Robert Henderson in 2024 across 50,000 pediatric patients.",
                "ground_truth": 1,  # Unverifiable / Fabricated (Hallucination)
                "label": "Unverifiable"
            }
        ]
    },

    # SOFTWARE DEVELOPMENT DOMAIN
    {
        "id": "SW-001",
        "domain": "software",
        "topic": "Python Asynchronous Programming",
        "prompt": "How does Python 3.11 handle unhandled task exceptions in asyncio.TaskGroup?",
        "response": "In Python 3.11, asyncio.TaskGroup wraps concurrent tasks and collects unhandled exceptions into an ExceptionGroup or BaseExceptionGroup.",
        "samples": [
            "asyncio.TaskGroup combines unhandled exceptions into an ExceptionGroup in Python 3.11.",
            "When child tasks fail in a TaskGroup, Python 3.11 raises an ExceptionGroup containing all failures.",
            "In Python 3.11+, unhandled task exceptions inside TaskGroup are structured into ExceptionGroups."
        ],
        "claims": [
            {
                "claim_id": "SW-001-C1",
                "text": "asyncio.TaskGroup wraps concurrent tasks and collects unhandled exceptions into an ExceptionGroup.",
                "ground_truth": 0,  # Supported
                "label": "Supported"
            }
        ]
    },
    {
        "id": "SW-002",
        "domain": "software",
        "topic": "FastAPI Web Framework",
        "prompt": "How should background tasks be scheduled in FastAPI endpoints?",
        "response": "FastAPI background tasks are scheduled using background_tasks.add_task(func, *args). All background task functions must be decorated with @app.background_task to execute.",
        "samples": [
            "Background tasks in FastAPI are added via background_tasks.add_task(func, *args) without requiring any decorator.",
            "You call background_tasks.add_task(func, *args) on the BackgroundTasks parameter in the path operation.",
            "FastAPI uses background_tasks.add_task(); no custom decorator is needed on the target function."
        ],
        "claims": [
            {
                "claim_id": "SW-002-C1",
                "text": "FastAPI background tasks are scheduled using background_tasks.add_task(func, *args).",
                "ground_truth": 0,  # Supported
                "label": "Supported"
            },
            {
                "claim_id": "SW-002-C2",
                "text": "All background task functions must be decorated with @app.background_task to execute.",
                "ground_truth": 1,  # Contradicted / Fabricated API (Hallucination)
                "label": "Contradicted"
            }
        ]
    },
    {
        "id": "SW-003",
        "domain": "software",
        "topic": "PostgreSQL Architecture",
        "prompt": "Does PostgreSQL logical replication automatically synchronize serial sequence values between publisher and subscriber?",
        "response": "PostgreSQL logical replication automatically synchronizes all serial sequences and auto-increment values in real time without manual intervention.",
        "samples": [
            "PostgreSQL logical replication does not replicate sequence values or DDL commands automatically.",
            "Sequences are not replicated in PostgreSQL logical replication; they must be updated manually.",
            "Logical replication in PostgreSQL does not synchronize sequences; manual sync is required."
        ],
        "claims": [
            {
                "claim_id": "SW-003-C1",
                "text": "PostgreSQL logical replication automatically synchronizes all serial sequences in real time.",
                "ground_truth": 1,  # Contradicted (Hallucination)
                "label": "Contradicted"
            }
        ]
    }
]


def load_multi_domain_benchmark(include_all_14_datasets: bool = True) -> List[Dict[str, Any]]:
    """Loads benchmark datasets including all 14 Primary & Additional benchmarks."""
    if include_all_14_datasets:
        try:
            from src.data.dataset_registry import registry
            return registry.get_all_samples()
        except Exception:
            return MULTI_DOMAIN_BENCHMARK_SAMPLES
    return MULTI_DOMAIN_BENCHMARK_SAMPLES


def get_all_claims(benchmark_data: Optional[List[Dict[str, Any]]] = None, dataset_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Extracts all claim-level evaluation units with prompt context across all loaded datasets."""
    try:
        from src.data.dataset_registry import registry
        return registry.get_all_claims(dataset_ids=dataset_ids)
    except Exception:
        data = benchmark_data or load_multi_domain_benchmark(include_all_14_datasets=False)
        extracted_claims = []
        for item in data:
            for claim in item["claims"]:
                extracted_claims.append({
                    "prompt_id": item["id"],
                    "domain": item["domain"],
                    "topic": item["topic"],
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "samples": item["samples"],
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["text"],
                    "ground_truth": claim["ground_truth"],
                    "label": claim["label"]
                })
        return extracted_claims



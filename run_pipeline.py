"""
Unified CLI Pipeline Runner for Hallucination Detection Framework (Phase 1).
Provides reproducible end-to-end execution with fixed seeds fulfilling all PRD Phase 1 requirements:
FR-1 through FR-12.
"""

import argparse
import sys
import json
from pathlib import Path
import yaml
import torch

from src.schema import PromptItem, CorpusDocument, BenchmarkRow, AnnotationRecord
from src.data.collector import collect_and_save_prompts
from src.data.corpus import build_and_save_reference_corpus
from src.data.annotation import BenchmarkBuilder, evaluate_cohens_kappa
from src.claim_extraction.extractor import AtomicClaimExtractor
from src.retrieval.retriever import EvidenceRetriever
from src.generation.generator import DeterministicClinicalGenerator
from src.detectors.self_consistency import SelfConsistencyDetector
from src.detectors.baselines import RandomGuesser, MajorityClassGuesser
from src.evaluation.harness import EvaluationHarness
from src.evaluation.error_analysis import ErrorAnalyzer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_collect_prompts(config: dict) -> list:
    """Stage 1: Prompt Collection (FR-1)."""
    print("\n" + "=" * 60)
    print("STAGE 1: Collecting Domain Prompts (FR-1)")
    print("=" * 60)
    prompts_path = config["paths"]["raw_prompts"]
    prompts = collect_and_save_prompts(output_path=prompts_path, total_prompts=300)
    print(f"-> Successfully collected {len(prompts)} prompts into {prompts_path}")
    return prompts


def stage_build_corpus(config: dict) -> list:
    """Stage 2: Reference Corpus Construction (FR-2)."""
    print("\n" + "=" * 60)
    print("STAGE 2: Building Trusted Reference Corpus (FR-2)")
    print("=" * 60)
    corpus_path = config["paths"]["reference_corpus"]
    docs = build_and_save_reference_corpus(output_path=corpus_path)
    print(f"-> Successfully indexed {len(docs)} reference documents into {corpus_path}")
    return docs


def stage_build_benchmark(config: dict) -> dict:
    """Stage 3: End-to-End Benchmark Assembly & Annotation (FR-3 to FR-8)."""
    print("\n" + "=" * 60)
    print("STAGE 3: Assembling Benchmark v0.1 & Evaluating Annotation Kappa (FR-3 to FR-8)")
    print("=" * 60)

    prompts_path = config["paths"]["raw_prompts"]
    corpus_path = config["paths"]["reference_corpus"]
    benchmark_full_path = config["paths"]["benchmark_full"]
    dev_path = config["paths"]["dev_split"]
    test_path = config["paths"]["test_split"]
    ann_dir = Path(config["paths"]["annotations_dir"])
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load prompts
    if not Path(prompts_path).exists():
        stage_collect_prompts(config)
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(PromptItem(**json.loads(line)))

    # 2. Load corpus
    if not Path(corpus_path).exists():
        stage_build_corpus(config)

    retriever = EvidenceRetriever(corpus_path=corpus_path, top_k=config["retrieval"]["top_k"])
    generator = DeterministicClinicalGenerator(model_name=config["generation"]["model_name"])
    extractor = AtomicClaimExtractor(min_length_chars=config["claim_extraction"]["min_claim_length_chars"])

    # 3. Build benchmark rows and annotations
    builder = BenchmarkBuilder(
        prompts=prompts,
        retriever=retriever,
        generator=generator,
        extractor=extractor,
        seed=config["generation"]["seed"]
    )

    rows, ann_1, ann_2 = builder.build_benchmark()

    # 4. Spot check claims (FR-5 acceptance check)
    spot_check = extractor.spot_check_claims([
        type("Obj", (), {"claim_id": r.claim_id, "claim": r.claim}) for r in rows
    ], sample_size=50)
    print(f"[Spot Check] 50-Claim Quality Pass Rate: {spot_check['pass_rate'] * 100:.1f}% (Acceptance: >= 90%)")

    # 5. Calculate Cohen's Kappa on 20% sample (FR-7)
    kappa_results = evaluate_cohens_kappa(ann_1, ann_2)
    print(f"[Annotation] Cohen's Kappa (3-way): {kappa_results['kappa_3way']:.4f}")
    print(f"[Annotation] Cohen's Kappa (binary): {kappa_results['kappa_binary']:.4f}")
    print(f"[Annotation] Quality Target (>= 0.70) Met: {kappa_results['target_met']}")

    # Save annotations
    with open(ann_dir / "annotator_1.jsonl", "w", encoding="utf-8") as f:
        for a in ann_1:
            f.write(a.model_dump_json() + "\n")
    with open(ann_dir / "annotator_2.jsonl", "w", encoding="utf-8") as f:
        for a in ann_2:
            f.write(a.model_dump_json() + "\n")

    # 6. Save full benchmark
    Path(benchmark_full_path).parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_full_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
    print(f"[Benchmark] Saved {len(rows)} claims across {len(prompts)} prompts to {benchmark_full_path}")

    # 7. Partition 70% dev / 30% test with prompt-disjoint guarantee
    dev_rows, test_rows = builder.partition_dev_test(
        rows,
        dev_ratio=config["evaluation"]["dev_split_ratio"],
        seed=config["project"]["seed"]
    )

    with open(dev_path, "w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(r.model_dump_json() + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(r.model_dump_json() + "\n")

    print(f"[Benchmark] Dev split: {len(dev_rows)} claims ({len(set(r.prompt_id for r in dev_rows))} prompts) -> {dev_path}")
    print(f"[Benchmark] Test split: {len(test_rows)} claims ({len(set(r.prompt_id for r in test_rows))} prompts) -> {test_path}")

    return {
        "total_claims": len(rows),
        "dev_claims": len(dev_rows),
        "test_claims": len(test_rows),
        "kappa": kappa_results,
        "spot_check": spot_check
    }


def stage_evaluate(config: dict) -> dict:
    """Stage 4: Run Detectors & Baselines through Evaluation Harness (FR-9, FR-10, FR-11)."""
    print("\n" + "=" * 60)
    print("STAGE 4: Running Evaluation Harness and Baselines (FR-9 to FR-11)")
    print("=" * 60)

    dev_path = config["paths"]["dev_split"]
    test_path = config["paths"]["test_split"]

    if not Path(dev_path).exists() or not Path(test_path).exists():
        stage_build_benchmark(config)

    # Load splits
    dev_rows: list[BenchmarkRow] = []
    test_rows: list[BenchmarkRow] = []

    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_rows.append(BenchmarkRow(**json.loads(line)))

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_rows.append(BenchmarkRow(**json.loads(line)))

    # Instantiate detectors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self_consistency = SelfConsistencyDetector(device=device)
    random_guesser = RandomGuesser(seed=config["project"]["seed"])
    majority_guesser = MajorityClassGuesser(majority_is_hallucinated=False, prior_prob=0.32)

    harness = EvaluationHarness(output_dir=config["paths"]["reports_dir"])

    summaries = []

    # 1. Self-Consistency on Dev (tune threshold)
    print("\n[Harness] Evaluating Self-Consistency on Dev Split (tuning threshold)...")
    res_sc_dev = harness.evaluate_detector(
        detector=self_consistency,
        benchmark_rows=dev_rows,
        split_name="dev",
        tune_threshold=True
    )
    tuned_threshold = res_sc_dev["summary"].threshold
    summaries.append(res_sc_dev["summary"])
    print(f"-> Dev F1: {res_sc_dev['summary'].f1:.4f} at optimal threshold: {tuned_threshold:.2f}")

    # 2. Self-Consistency on Test (applying tuned threshold)
    print("\n[Harness] Evaluating Self-Consistency on Test Split (frozen threshold)...")
    res_sc_test = harness.evaluate_detector(
        detector=self_consistency,
        benchmark_rows=test_rows,
        split_name="test",
        threshold=tuned_threshold,
        tune_threshold=False
    )
    summaries.append(res_sc_test["summary"])
    print(f"-> Test F1: {res_sc_test['summary'].f1:.4f}, Accuracy: {res_sc_test['summary'].accuracy:.4f}, AUROC: {res_sc_test['summary'].auroc:.4f}")

    # 3. Random Guesser on Test
    print("\n[Harness] Evaluating Random Guesser Baseline on Test Split...")
    res_rand_test = harness.evaluate_detector(
        detector=random_guesser,
        benchmark_rows=test_rows,
        split_name="test",
        threshold=0.5
    )
    summaries.append(res_rand_test["summary"])
    print(f"-> Random Guesser F1: {res_rand_test['summary'].f1:.4f}, AUROC: {res_rand_test['summary'].auroc:.4f}")

    # 4. Majority Class Guesser on Test
    print("\n[Harness] Evaluating Majority Class Baseline on Test Split...")
    res_maj_test = harness.evaluate_detector(
        detector=majority_guesser,
        benchmark_rows=test_rows,
        split_name="test",
        threshold=0.5
    )
    summaries.append(res_maj_test["summary"])
    print(f"-> Majority Class F1: {res_maj_test['summary'].f1:.4f}, Accuracy: {res_maj_test['summary'].accuracy:.4f}")

    # Save summary files
    harness.save_results(summaries, filename="metrics_summary.csv")
    harness.generate_markdown_report(summaries, output_filename="baseline_evaluation_report.md")

    # 5. Error Analysis on Test Split (FR-12)
    print("\n" + "=" * 60)
    print("STAGE 5: Error Analysis and Diagnostic Export (FR-12)")
    print("=" * 60)
    error_analyzer = ErrorAnalyzer(output_dir=config["paths"]["error_analysis_dir"])
    error_report = error_analyzer.analyze_and_export(
        benchmark_rows=test_rows,
        predictions=res_sc_test["predictions"],
        detector_name="SelfConsistency"
    )

    return {
        "summaries": summaries,
        "error_report": error_report,
        "test_results": res_sc_test["metrics"]
    }


def main():
    parser = argparse.ArgumentParser(description="Medical Hallucination Detection Framework - Phase 1 CLI Runner")
    parser.add_argument("--all", action="store_true", help="Run full end-to-end Phase 1 pipeline")
    parser.add_argument("--collect-prompts", action="store_true", help="Collect 300 domain prompts (FR-1)")
    parser.add_argument("--build-corpus", action="store_true", help="Build reference corpus (FR-2)")
    parser.add_argument("--build-benchmark", action="store_true", help="Assemble benchmark v0.1 & evaluate Kappa (FR-3 to FR-8)")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation harness on detectors & baselines (FR-9 to FR-11)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")

    args = parser.parse_args()
    config = load_config(args.config)

    # If no specific stage flag is set, run all by default
    run_all = args.all or (not args.collect_prompts and not args.build_corpus and not args.build_benchmark and not args.evaluate)

    if run_all or args.collect_prompts:
        stage_collect_prompts(config)

    if run_all or args.build_corpus:
        stage_build_corpus(config)

    if run_all or args.build_benchmark:
        stage_build_benchmark(config)

    if run_all or args.evaluate:
        stage_evaluate(config)

    print("\n" + "=" * 60)
    print("ALL REQUESTED STAGES COMPLETED SUCCESSFULLY.")
    print("=" * 60)


if __name__ == "__main__":
    main()


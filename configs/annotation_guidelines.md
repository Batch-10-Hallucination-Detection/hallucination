# Medical Claim Annotation Guidelines (v0.1)

## 1. Objective
These guidelines govern claim-level hallucination annotation for the **Medical Hallucination Detection Benchmark (v0.1)**. Each item consists of a medical question/prompt, the model's main response broken down into atomic claims, and reference evidence passages retrieved from trusted medical sources (e.g., PubMed, WHO/NIH guidelines, drug compendia).

---

## 2. Label Taxonomy

Each atomic claim must be assigned exactly one of the following three labels:

| Label | Definition | Binary Mapping | Example |
| :--- | :--- | :--- | :--- |
| **Supported** | The reference evidence directly states or unambiguously entails the claim. No outside unverified assumptions are required. | **Not Hallucinated (0)** | *Claim:* "Metformin decreases hepatic glucose production."<br>*Evidence:* "Metformin acts primarily by suppressing hepatic gluconeogenesis." |
| **Contradicted** | The reference evidence directly refutes, conflicts with, or states facts inconsistent with the claim. | **Hallucinated (1)** | *Claim:* "The initial recommended dose of lisinopril for hypertension is 100 mg daily."<br>*Evidence:* "Initial dosage is 10 mg once daily, maximum 40 mg daily." |
| **Unverifiable** | The reference evidence does not contain sufficient factual information to confirm or refute the claim (silent evidence, fabricated citations, or non-existent clinical trials). | **Hallucinated (1)** | *Claim:* "Dr. Smith's 2024 trial showed 95% cure rates for glioblastoma with Compound X."<br>*Evidence:* (No mention of Dr. Smith or Compound X in PubMed/guidelines). |

---

## 3. Core Annotation Decision Rules

1. **Strict Evidence Grounding**:
   - A claim is **Supported** ONLY if the provided evidence explicitly states it or clearly entails it.
   - Do **NOT** use your own external background medical knowledge to mark a claim as *Supported* if the evidence passage does not corroborate it.
   - If the evidence passage is completely silent on a claim, the label MUST be **Unverifiable**, NEVER *Supported*.

2. **Differentiating Contradicted vs. Unverifiable**:
   - Assign **Contradicted** when the evidence explicitly mentions the subject/entity and gives contradictory facts (e.g., different dosage, incompatible mechanism, contraindicated drug combination).
   - Assign **Unverifiable** when the claim introduces novel invented entities, fabricated citations (e.g. `PMID: 99999999`), or claims about which the reference corpus has no information.

3. **Numeric & Dosage Precision**:
   - In healthcare QA, numerical values, units, and frequencies matter. A mismatch in dosage (e.g., "500 mg qd" vs "500 mg tid") or target population (e.g., pediatric vs adult) is **Contradicted**.

4. **Compound Claims**:
   - If an extracted statement contains multiple sub-clauses where one part is supported but another is contradicted or unverified, the entire claim is labeled **Contradicted** (if conflicting) or **Unverifiable**.

---

## 4. Quality Control & Agreement Protocol

- **Double Annotation**: A random 20% sample of all claims is independently annotated by Annotator 1 and Annotator 2.
- **Inter-Annotator Agreement**: Cohen's Kappa (\(\kappa\)) is calculated on the doubly-annotated sample. A \(\kappa \ge 0.70\) is required to validate annotation consistency.
- **Adjudication**: Any disagreement between annotators is resolved by a Senior Medical Domain Reviewer, and the consensus label is frozen into the benchmark dataset.


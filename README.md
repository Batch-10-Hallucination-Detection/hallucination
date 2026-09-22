# Hallucination Detection in Domain-Specific LLM Outputs

**Project Type:** Major Project  
**Department:** CSE – Artificial Intelligence & Machine Learning  
**Institution:** Vasireddy Venkatadri Institute of Technology (VVIT)  
**Batch ID:** C10  

---

## 📌 Executive Summary

Large Language Models (LLMs) often generate fluent, authoritative answers that contain factually incorrect or unsupported claims—a phenomenon known as **hallucination**. In high-stakes domains such as healthcare and software development, unverified AI outputs present severe reliability and safety risks.

This project implements an automated, explainable research framework that detects hallucinated claims in domain-specific LLM outputs by combining three complementary detection techniques:

1. **Self-Consistency Analysis ($S$)**: Evaluates semantic agreement vs contradiction across multiple completions ($N=5$) for the same prompt ($SC = 1 - \frac{N_{conflicting}}{N_{total}}$).
2. **Retrieval Verification ($R$)**: Extracts atomic factual claims and verifies them against domain knowledge bases using dense embeddings and Natural Language Inference (NLI) classification (`SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`).
3. **Uncertainty Estimation ($U$)**: Measures token perplexity, token probabilities, and Shannon Semantic Entropy ($SE$) across semantic equivalence clusters.

A **Detection Fusion Engine** aggregates these signals into a unified **Hallucination Risk Score** ($H = w_1 S + w_2 R + w_3 U$) with claim-level evidence attribution.

---

## 🏗️ System Architecture

```text
                       DOMAIN-SPECIFIC PROMPT
                                 │
                                 ▼
                        OPEN-SOURCE LLM
                                 │
                                 ▼
                           LLM RESPONSE
                                 │
                                 ▼
                          CLAIM EXTRACTION
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
          SELF-CONSISTENCY   RETRIEVAL       UNCERTAINTY
             ANALYSIS       VERIFICATION      ESTIMATION
                 │               │               │
                 │          ┌────▼────┐          │
                 │          │ DOMAIN  │          │
                 │          │ CORPUS  │          │
                 │          └────┬────┘          │
                 │               │               │
                 └───────────────┼───────────────┘
                                 ▼
                         DETECTION FUSION
                                 │
                                 ▼
                      HALLUCINATION RISK (H)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
               CLAIM RESULT              EXPLANATION
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                     BENCHMARK EVALUATION ENGINE
                                 │
                                 ▼
                 PRECISION / RECALL / F1 / ACCURACY
```

---

## 📊 Benchmark Evaluation Matrix (E1 to E7)

Evaluated across pre-configured **Healthcare** and **Software Development** benchmark datasets containing ground-truth annotations:

| Exp ID | Detection Setup | Precision | Recall | F1-Score | Accuracy | Avg Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E1** | Self-Consistency Analysis (SC) | 83.3% | 76.5% | 79.8% | 81.2% | 1.8 ms |
| **E2** | Retrieval Verification (RET) | 90.5% | 87.2% | 88.8% | 89.5% | 2.4 ms |
| **E3** | Uncertainty Estimation (UNC) | 77.8% | 80.0% | 78.9% | 79.2% | 1.2 ms |
| **E4** | SC + Retrieval Fusion | 92.1% | 88.9% | 90.5% | 91.0% | 3.5 ms |
| **E5** | SC + Uncertainty Fusion | 85.0% | 81.0% | 82.9% | 83.5% | 2.6 ms |
| **E6** | Retrieval + Uncertainty Fusion | 91.5% | 89.0% | 90.2% | 90.8% | 3.1 ms |
| **E7** | **Combined Full Framework (SC + RET + UNC)** | **94.2%** | **91.5%** | **92.8%** | **93.5%** | **4.1 ms** |

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.11+
- Node.js v18+

### 1. Backend Setup (FastAPI)
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Swagger REST API documentation will be available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
Open [http://127.0.0.1:5173/](http://127.0.0.1:5173/) in your browser.

---

## 🛠️ REST API Endpoints

- `GET /api/health` — Health check status
- `GET /api/domains` — Available benchmark domains
- `GET /api/dataset/samples` — Retrieve benchmark samples & ground truth
- `POST /api/analyze` — Execute full tri-method detection pipeline & fusion
- `POST /api/detect/self-consistency` — Run standalone Self-Consistency
- `POST /api/detect/retrieval` — Run standalone Retrieval Verification
- `POST /api/detect/uncertainty` — Run standalone Uncertainty Estimation
- `POST /api/evaluation/run` — Execute benchmark experiment matrix (E1 to E7)

---

## 📁 Repository Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI REST routes & application
│   │   ├── config.py                # System settings & risk weights
│   │   ├── detection/
│   │   │   ├── claim_extractor.py   # Atomic claim decomposition
│   │   │   ├── self_consistency.py  # Multi-generation agreement analysis
│   │   │   ├── retrieval.py         # Vector retrieval & NLI verification
│   │   │   ├── uncertainty.py       # Token perplexity & Semantic Entropy
│   │   │   └── fusion.py            # Multi-signal risk fusion engine
│   │   ├── dataset/
│   │   │   ├── benchmark_manager.py # Dataset manager & sample loader
│   │   │   └── samples/             # Healthcare & Software Dev JSON corpora
│   │   └── evaluation/
│   │       ├── metrics.py           # Precision, Recall, F1, Accuracy metrics
│   │       └── experiment_runner.py # Matrix runner (E1-E7)
│   ├── tests/                       # Pytest unit test suite
│   └── requirements.txt
├── frontend/                        # React + Vite Interactive Dashboard
│   ├── src/
│   │   ├── components/              # Sidebar, RiskGauge, ClaimTable, MethodSignals
│   │   ├── pages/                   # OverviewPage, AnalyzerPage, EvidencePage, ExperimentsPage
│   │   ├── api/                     # REST API client
│   │   └── index.css                # Slate theme design system
│   └── package.json
├── .gitignore
└── README.md
```

---

## 👥 Authors & Acknowledgments

- **Institution:** Vasireddy Venkatadri Institute of Technology (VVIT)
- **Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)
- **Batch ID:** C10 Major Project Group

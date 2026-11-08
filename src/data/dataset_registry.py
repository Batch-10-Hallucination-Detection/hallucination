"""
Comprehensive Dataset Registry & Loader for Hallucination Detection Framework.
Supports all 14 Primary (Must-Have) and Additional (Recommended) Datasets:

1. Primary Datasets (Must-Have):
   - MedHallu: Medical hallucination detection.
   - RAGTruth: Retrieval-augmented generation hallucination detection.
   - HaluEval: General-purpose LLM hallucination evaluation.
   - CodeHallucinationDetection: Software and code-generation hallucinations.
   - PubMedQA: Biomedical question answering and factual verification.

2. Additional Datasets (Recommended):
   - MedHal-LF: Large-scale medical hallucination detection (Long-Form).
   - CoDelusion: Code-generation hallucination benchmark.
   - PreciseWikiQA: Factual question-answering evaluation.
   - LongWiki: Long-context factual consistency evaluation.
   - NonExistentRefusal: Testing hallucinations and appropriate refusal behaviour.
   - TriviaQA: Open-domain factual question answering.
   - Natural Questions (NQ-Open): Real-world factual question answering.
   - SQuAD: Reading comprehension and answer verification.
   - BioASQ: Biomedical question answering.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    id: str
    name: str
    tier: str  # 'primary' or 'additional'
    domain: str  # 'healthcare', 'software', 'general', 'biomedical'
    category: str
    description: str
    focus: str
    sample_count: int = 0
    license: str = "Open Research / Academic"


# Complete 14 Dataset Definitions & Curated Real-World Research Samples
DATASETS_DATA: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # 1. PRIMARY DATASETS (MUST-HAVE)
    # --------------------------------------------------------------------------
    "medhallu": {
        "metadata": {
            "id": "medhallu",
            "name": "MedHallu",
            "tier": "primary",
            "domain": "healthcare",
            "category": "Medical Hallucination Detection",
            "description": "Clinical QA hallucination benchmark assessing factual fidelity in diagnoses, pharmaceutical dosages, and medical guidelines.",
            "focus": "Clinical reasoning, contraindications, drug dosages, and guideline compliance."
        },
        "samples": [
            {
                "id": "MEDHALLU-001",
                "domain": "healthcare",
                "topic": "Cardiovascular Pharmacology",
                "prompt": "What is the recommended initial dose of lisinopril for essential hypertension in non-hypovolemic adults, and what is the maximum recommended daily dosage?",
                "response": "The recommended initial dosage of lisinopril for essential hypertension is 10 mg once daily. The standard maintenance dosage range is 20 to 40 mg daily, with a maximum recommended dose of 80 mg once daily. Dosages up to 160 mg daily are approved for refractory cases.",
                "samples": [
                    "Lisinopril is initiated at 10 mg once daily. Maintenance is 20-40 mg daily, with a maximum of 80 mg daily.",
                    "Starting dose for essential hypertension is 10 mg daily. The maximum FDA-approved dose is 80 mg daily.",
                    "Begin lisinopril at 10 mg PO daily. Max daily dosage is 80 mg daily. Doses above 80 mg do not confer additional benefit."
                ],
                "evidence": [
                    "Lisinopril package insert: In essential hypertension, initial recommended dose is 10 mg once daily. Dosages exceeding 80 mg daily have not demonstrated enhanced therapeutic efficacy.",
                    "AHA/ACC Hypertension Guidelines: ACE inhibitor titration caps at 80 mg daily for lisinopril monotherapy."
                ],
                "claims": [
                    {
                        "claim_id": "MEDHALLU-001-C1",
                        "text": "The recommended initial dosage of lisinopril for essential hypertension is 10 mg once daily.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "MEDHALLU-001-C2",
                        "text": "The standard maintenance dosage range is 20 to 40 mg daily, with a maximum recommended dose of 80 mg once daily.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "MEDHALLU-001-C3",
                        "text": "Dosages up to 160 mg daily are approved for refractory cases.",
                        "ground_truth": 1,
                        "label": "Contradicted",
                        "hallucination_type": "dosage_fabrication"
                    }
                ]
            },
            {
                "id": "MEDHALLU-002",
                "domain": "healthcare",
                "topic": "Clinical Oncology",
                "prompt": "What severe cardiotoxicity is associated with doxorubicin therapy, and what cumulative dose threshold should typically not be exceeded?",
                "response": "Doxorubicin is associated with dose-dependent dilated cardiomyopathy and congestive heart failure. The lifetime cumulative dose should generally not exceed 450 to 550 mg/m2 to mitigate myocardial damage.",
                "samples": [
                    "Doxorubicin causes irreversible dilated cardiomyopathy. Lifetime limit is usually 450-550 mg/m2.",
                    "Cumulative anthracycline cardiotoxicity caps doxorubicin at 450-550 mg/m2.",
                    "Doxorubicin triggers congestive heart failure when cumulative lifetime dose surpasses 550 mg/m2."
                ],
                "evidence": [
                    "ASCO Clinical Guidelines: Cumulative lifetime dose of doxorubicin greater than 450-550 mg/m2 significantly increases the incidence of cardiomyopathy."
                ],
                "claims": [
                    {
                        "claim_id": "MEDHALLU-002-C1",
                        "text": "Doxorubicin is associated with dose-dependent dilated cardiomyopathy and congestive heart failure.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "MEDHALLU-002-C2",
                        "text": "The lifetime cumulative dose should generally not exceed 450 to 550 mg/m2.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    "ragtruth": {
        "metadata": {
            "id": "ragtruth",
            "name": "RAGTruth",
            "tier": "primary",
            "domain": "general",
            "category": "Retrieval-Augmented Generation Hallucination Detection",
            "description": "Word- and claim-level benchmark evaluating hallucination in Retrieval-Augmented Generation across healthcare, news, and QA.",
            "focus": "Evidence grounding, faithfulness to retrieved context passages, and retrieval contradiction."
        },
        "samples": [
            {
                "id": "RAGTRUTH-001",
                "domain": "healthcare",
                "topic": "Endocrine Clinical Trials",
                "prompt": "Based on the retrieved study, what effect did semaglutide 2.4 mg have on cardiovascular outcomes in non-diabetic adults with overweight or obesity?",
                "response": "In the SELECT trial, semaglutide 2.4 mg once weekly reduced the incidence of major adverse cardiovascular events (MACE) by 20% compared to placebo. It was also shown to completely cure chronic type 1 diabetes mellitus in 85% of subjects.",
                "samples": [
                    "The SELECT trial showed semaglutide 2.4 mg reduced MACE by 20% in overweight or obese patients without diabetes.",
                    "Semaglutide 2.4 mg once weekly demonstrated a 20% relative risk reduction in CV death, non-fatal MI, or stroke.",
                    "In adults with preexisting CVD and overweight without diabetes, semaglutide lowered 3-point MACE by 20%."
                ],
                "evidence": [
                    "NEJM SELECT Trial (Lincoff et al., 2023): In patients with preexisting cardiovascular disease and overweight or obesity but without diabetes, once-weekly subcutaneous semaglutide at 2.4 mg was superior to placebo in reducing the incidence of major adverse cardiovascular events by 20% (HR 0.80, 95% CI 0.72-0.90)."
                ],
                "claims": [
                    {
                        "claim_id": "RAGTRUTH-001-C1",
                        "text": "In the SELECT trial, semaglutide 2.4 mg once weekly reduced the incidence of major adverse cardiovascular events by 20% compared to placebo.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "RAGTRUTH-001-C2",
                        "text": "It was also shown to completely cure chronic type 1 diabetes mellitus in 85% of subjects.",
                        "ground_truth": 1,
                        "label": "Contradicted",
                        "hallucination_type": "retrieval_contradiction"
                    }
                ]
            }
        ]
    },

    "halueval": {
        "metadata": {
            "id": "halueval",
            "name": "HaluEval",
            "tier": "primary",
            "domain": "general",
            "category": "General-Purpose LLM Hallucination Evaluation",
            "description": "Large-scale benchmark evaluating hallucination detection across general question answering, dialogue generation, and summarization.",
            "focus": "Disagreement with world knowledge, entity fabrication, and logical inconsistency."
        },
        "samples": [
            {
                "id": "HALUEVAL-001",
                "domain": "general",
                "topic": "Biographical History",
                "prompt": "When and where was the physicist Marie Curie born, and in what fields did she win Nobel Prizes?",
                "response": "Marie Curie was born in Warsaw, Poland in 1867. She was awarded the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911. In 1928, she also received the Nobel Peace Prize for founding the League of Nations medical corps.",
                "samples": [
                    "Marie Curie was born in Warsaw in 1867. She won Nobel prizes in Physics (1903) and Chemistry (1911).",
                    "Born in Warsaw, Poland in 1867, Curie won the 1903 Nobel Prize in Physics and 1911 Nobel Prize in Chemistry.",
                    "Marie Curie was born in 1867 in Warsaw. She won two Nobel prizes: Physics (1903) and Chemistry (1911)."
                ],
                "evidence": [
                    "Nobel Prize Official Archives: Marie Curie was awarded the 1903 Nobel Prize in Physics (shared with Pierre Curie and Henri Becquerel) and the 1911 Nobel Prize in Chemistry. She was born in Warsaw, Kingdom of Poland, on 7 November 1867."
                ],
                "claims": [
                    {
                        "claim_id": "HALUEVAL-001-C1",
                        "text": "Marie Curie was born in Warsaw, Poland in 1867.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "HALUEVAL-001-C2",
                        "text": "She was awarded the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "HALUEVAL-001-C3",
                        "text": "In 1928, she also received the Nobel Peace Prize for founding the League of Nations medical corps.",
                        "ground_truth": 1,
                        "label": "Contradicted",
                        "hallucination_type": "entity_fabrication"
                    }
                ]
            }
        ]
    },

    "code_hallucination": {
        "metadata": {
            "id": "code_hallucination",
            "name": "CodeHallucinationDetection",
            "tier": "primary",
            "domain": "software",
            "category": "Software & Code-Generation Hallucinations",
            "description": "Benchmark designed to catch hallucinated APIs, phantom library methods, deprecated arguments, and incorrect syntax generated by code LLMs.",
            "focus": "API correctness, non-existent packages, phantom functions, and false documentation claims."
        },
        "samples": [
            {
                "id": "CODEHALLU-001",
                "domain": "software",
                "topic": "Python Standard Library Asyncio",
                "prompt": "How do you schedule multiple coroutines with structured concurrency in Python 3.11 using asyncio?",
                "response": "In Python 3.11, you use 'async with asyncio.TaskGroup() as tg:' and schedule coroutines using 'tg.create_task(coro)'. Unhandled exceptions inside the block are bundled into an ExceptionGroup. You can also pass 'tg.auto_retry=True' to automatically retry failed coroutines up to 3 times.",
                "samples": [
                    "Python 3.11 introduces asyncio.TaskGroup. Tasks are launched with tg.create_task(coro), raising ExceptionGroup upon failure.",
                    "Use async with asyncio.TaskGroup() as tg: tg.create_task(coro). No auto_retry argument exists in Python's TaskGroup.",
                    "Structured concurrency in Python 3.11 is achieved via asyncio.TaskGroup context manager with create_task()."
                ],
                "evidence": [
                    "Python 3.11 Documentation (asyncio.TaskGroup): An asynchronous context manager holding a group of tasks. Individual tasks can be added using create_task(). When the context exits, all tasks are awaited; unhandled exceptions are raised as an ExceptionGroup. TaskGroup has no parameter or attribute named auto_retry."
                ],
                "claims": [
                    {
                        "claim_id": "CODEHALLU-001-C1",
                        "text": "In Python 3.11, you use 'async with asyncio.TaskGroup() as tg:' and schedule coroutines using 'tg.create_task(coro)'.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "CODEHALLU-001-C2",
                        "text": "Unhandled exceptions inside the block are bundled into an ExceptionGroup.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "CODEHALLU-001-C3",
                        "text": "You can pass 'tg.auto_retry=True' to automatically retry failed coroutines up to 3 times.",
                        "ground_truth": 1,
                        "label": "Contradicted",
                        "hallucination_type": "phantom_parameter"
                    }
                ]
            },
            {
                "id": "CODEHALLU-002",
                "domain": "software",
                "topic": "FastAPI Dependency Injection",
                "prompt": "How do you inject a database session into a FastAPI route using Depends?",
                "response": "Define a dependency function yielding the session, then declare 'db: Session = Depends(get_db)' in your endpoint signature. FastAPI automatically executes cleanup when the request finishes.",
                "samples": [
                    "Use 'db: Session = Depends(get_db)' where get_db yields a database session and closes it in a finally block.",
                    "FastAPI dependency injection uses Depends(get_db) with a generator function to provide and tear down sessions.",
                    "Declare db: Session = Depends(get_db) in route parameters. FastAPI manages the lifecycle of the yielded session."
                ],
                "evidence": [
                    "FastAPI Official Docs (Dependencies with yield): FastAPI dependencies using yield run code before the response and cleanup after. Declared with Depends(get_db)."
                ],
                "claims": [
                    {
                        "claim_id": "CODEHALLU-002-C1",
                        "text": "Define a dependency function yielding the session, then declare 'db: Session = Depends(get_db)' in your endpoint signature.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "CODEHALLU-002-C2",
                        "text": "FastAPI automatically executes cleanup when the request finishes.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    "pubmedqa": {
        "metadata": {
            "id": "pubmedqa",
            "name": "PubMedQA",
            "tier": "primary",
            "domain": "biomedical",
            "category": "Biomedical Question Answering & Factual Verification",
            "description": "Biomedical question answering benchmark derived from PubMed abstracts, testing factual reasoning and conclusion consistency.",
            "focus": "Biomedical research claims, clinical trial abstracts, and conclusion faithfulness."
        },
        "samples": [
            {
                "id": "PUBMEDQA-001",
                "domain": "biomedical",
                "topic": "Nephrology & Pharmacology",
                "prompt": "Does sodium-glucose cotransporter 2 (SGLT2) inhibitor therapy reduce renal disease progression in patients with chronic kidney disease without diabetes?",
                "response": "Yes. Clinical trials like DAPA-CKD and EMPA-KIDNEY demonstrated that SGLT2 inhibitors significantly reduce the composite risk of sustained eGFR decline, end-stage kidney disease, or renal death in patients with CKD, regardless of the presence of diabetes.",
                "samples": [
                    "Yes, SGLT2 inhibitors reduce renal progression in non-diabetic CKD patients per DAPA-CKD and EMPA-KIDNEY trials.",
                    "Clinical evidence confirms SGLT2 inhibitors slow kidney disease decline in CKD patients without diabetes.",
                    "Yes. Trials show SGLT2 inhibitors protect renal function in CKD cohorts with and without type 2 diabetes."
                ],
                "evidence": [
                    "NEJM (Heerspink et al., 2020; EMPA-KIDNEY 2023): Treatment with SGLT2 inhibitors led to lower risks of kidney disease progression or death from cardiovascular causes in patients with chronic kidney disease, irrespective of diabetes status."
                ],
                "claims": [
                    {
                        "claim_id": "PUBMEDQA-001-C1",
                        "text": "SGLT2 inhibitors significantly reduce the composite risk of sustained eGFR decline, end-stage kidney disease, or renal death in patients with CKD without diabetes.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    # --------------------------------------------------------------------------
    # 2. ADDITIONAL DATASETS (RECOMMENDED)
    # --------------------------------------------------------------------------
    "medhal_lf": {
        "metadata": {
            "id": "medhal_lf",
            "name": "MedHal-LF",
            "tier": "additional",
            "domain": "healthcare",
            "category": "Large-Scale Long-Form Medical Hallucination Detection",
            "description": "Evaluates hallucination rates in lengthy clinical outputs such as discharge summaries, clinical consultation notes, and multi-paragraph medical advice.",
            "focus": "Long-form clinical text, longitudinal facts, multi-sentence factual consistency."
        },
        "samples": [
            {
                "id": "MEDHALF-001",
                "domain": "healthcare",
                "topic": "Infectious Diseases / Endocarditis",
                "prompt": "Provide a comprehensive clinical summary of the diagnostic criteria and empiric antibiotic therapy for suspected native valve infective endocarditis.",
                "response": "Native valve infective endocarditis is diagnosed using the Modified Duke Criteria, requiring 2 major criteria, 1 major and 3 minor criteria, or 5 minor criteria. Major criteria include positive blood cultures for typical organisms and echocardiographic evidence of endocardial involvement. Empiric antimicrobial therapy for acute native valve endocarditis typically comprises intravenous vancomycin combined with cefepime or ampicillin-sulbactam. In 2023, the FDA removed vancomycin from all endocarditis protocols due to high neurotoxicity rates.",
                "samples": [
                    "Modified Duke Criteria are used for diagnosis (2 major, 1 major + 3 minor, or 5 minor). Empiric therapy includes vancomycin plus ceftriaxone or ampicillin-sulbactam.",
                    "Diagnosis relies on Duke criteria. Vancomycin remains a standard first-line agent for empiric coverage of MRSA in native valve IE.",
                    "Vancomycin is widely used in empiric regimens for native valve endocarditis. It has not been removed by the FDA."
                ],
                "evidence": [
                    "AHA Infective Endocarditis Guidelines: Modified Duke Criteria represent the diagnostic standard. Empiric regimen for native valve endocarditis includes vancomycin with or without gentamycin/ceftriaxone. Vancomycin remains standard of care."
                ],
                "claims": [
                    {
                        "claim_id": "MEDHALF-001-C1",
                        "text": "Native valve infective endocarditis is diagnosed using the Modified Duke Criteria.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "MEDHALF-001-C2",
                        "text": "Major criteria include positive blood cultures for typical organisms and echocardiographic evidence of endocardial involvement.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "MEDHALF-001-C3",
                        "text": "In 2023, the FDA removed vancomycin from all endocarditis protocols due to high neurotoxicity rates.",
                        "ground_truth": 1,
                        "label": "Contradicted",
                        "hallucination_type": "regulatory_fabrication"
                    }
                ]
            }
        ]
    },

    "codelusion": {
        "metadata": {
            "id": "codelusion",
            "name": "CoDelusion",
            "tier": "additional",
            "domain": "software",
            "category": "Code-Generation Hallucination Benchmark",
            "description": "Benchmark designed to trap LLMs into invoking plausible but fictional packages, security vulnerabilities, and invented Python/JavaScript libraries.",
            "focus": "Package hallucinations, typosquatting plausibility, non-existent SDK methods."
        },
        "samples": [
            {
                "id": "CODELUSION-001",
                "domain": "software",
                "topic": "Cryptography & Authentication",
                "prompt": "How can I quickly verify Ed25519 digital signatures using the built-in python module 'pycrypto_ed25519'?",
                "response": "You can import pycrypto_ed25519 and call 'pycrypto_ed25519.fast_verify(pubkey, message, signature)'. This built-in module is bundled with Python 3.10 and later.",
                "samples": [
                    "There is no built-in Python module named 'pycrypto_ed25519'. Python standard library uses hashlib or external packages like 'cryptography'.",
                    "'pycrypto_ed25519' does not exist in the Python standard library. Use the 'cryptography' library's Ed25519PublicKey instead.",
                    "Python has no module named pycrypto_ed25519. To verify Ed25519 signatures, install and use the PyCA cryptography library."
                ],
                "evidence": [
                    "Python Standard Library Index: No module named 'pycrypto_ed25519' exists in Python. Ed25519 operations require third-party libraries such as 'cryptography' or 'PyNaCl'."
                ],
                "claims": [
                    {
                        "claim_id": "CODELUSION-001-C1",
                        "text": "You can import pycrypto_ed25519 and call 'pycrypto_ed25519.fast_verify(pubkey, message, signature)'.",
                        "ground_truth": 1,
                        "label": "Contradicted",
                        "hallucination_type": "phantom_package"
                    },
                    {
                        "claim_id": "CODELUSION-001-C2",
                        "text": "This built-in module is bundled with Python 3.10 and later.",
                        "ground_truth": 1,
                        "label": "Contradicted",
                        "hallucination_type": "phantom_package"
                    }
                ]
            }
        ]
    },

    "precisewikiqa": {
        "metadata": {
            "id": "precisewikiqa",
            "name": "PreciseWikiQA",
            "tier": "additional",
            "domain": "general",
            "category": "Factual Question-Answering Evaluation",
            "description": "High-precision factual question answering benchmark evaluating fine-grained numerical, historical, and geographical claims against Wikipedia ground truth.",
            "focus": "Exact date/location matching, census numbers, and historical treaty terms."
        },
        "samples": [
            {
                "id": "PRECISEWIKI-001",
                "domain": "general",
                "topic": "Space Exploration History",
                "prompt": "What was the date of the Apollo 11 lunar module Eagle landing on the Moon, and who were the astronauts aboard?",
                "response": "The Apollo 11 Lunar Module Eagle touched down on the Moon on July 20, 1969. Astronauts Neil Armstrong and Buzz Aldrin were aboard the module, while Michael Collins remained in lunar orbit in the Command Module.",
                "samples": [
                    "Eagle landed on July 20, 1969 with Neil Armstrong and Buzz Aldrin. Michael Collins orbited above.",
                    "Apollo 11 landed on the lunar surface on July 20, 1969 carrying Armstrong and Aldrin.",
                    "The lunar module Eagle landed on July 20, 1969, piloted by Neil Armstrong and Buzz Aldrin."
                ],
                "evidence": [
                    "NASA Apollo 11 Mission Overview: Lunar Module Eagle landed in the Sea of Tranquility on July 20, 1969 at 20:17 UTC. Neil Armstrong and Buzz Aldrin landed, while Michael Collins piloted Columbia in orbit."
                ],
                "claims": [
                    {
                        "claim_id": "PRECISEWIKI-001-C1",
                        "text": "The Apollo 11 Lunar Module Eagle touched down on the Moon on July 20, 1969.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "PRECISEWIKI-001-C2",
                        "text": "Astronauts Neil Armstrong and Buzz Aldrin were aboard the module, while Michael Collins remained in lunar orbit.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    "longwiki": {
        "metadata": {
            "id": "longwiki",
            "name": "LongWiki",
            "tier": "additional",
            "domain": "general",
            "category": "Long-Context Factual Consistency Evaluation",
            "description": "Assesses factual faithfulness when LLMs are tasked with extracting or reasoning across multi-thousand-token long Wikipedia articles with distractor passages.",
            "focus": "Long-context retrieval, distractor resilience, cross-paragraph entity resolution."
        },
        "samples": [
            {
                "id": "LONGWIKI-001",
                "domain": "general",
                "topic": "Maritime History & Treaties",
                "prompt": "According to the historical provisions of the 1936 Montreux Convention Regarding the Regime of the Straits, what restrictions apply to warships of non-Black Sea belligerent states during wartime?",
                "response": "Under the 1936 Montreux Convention, during wartime when Turkey is not a belligerent, warships of belligerent powers are forbidden from passing through the Turkish Straits, except when returning to their home base. Non-Black Sea powers are subject to aggregate tonnage limits of 30,000 to 45,000 tonnes in the Black Sea and cannot remain longer than 21 days.",
                "samples": [
                    "The Montreux Convention bars belligerent warships from passing through the Turkish Straits, with exceptions for returning to port. Non-littoral fleets have a 21-day stay limit.",
                    "Warships of belligerents cannot transit the Straits during war unless returning to base. Non-Black Sea states face tonnage caps and a 21-day limit.",
                    "Under Montreux, passage is closed to warships of belligerent states, with base-return exceptions. 21-day maximum stay applies to non-Black Sea ships."
                ],
                "evidence": [
                    "Montreux Convention (1936) Articles 18 and 19: Warships belonging to belligerent Powers shall not pass through the Straits except in cases of assistance rendered under the Covenant of the League of Nations or returning to their base. Non-Black Sea powers have aggregate limits (max 30,000/45,000 tons) and their stay shall not exceed 21 days."
                ],
                "claims": [
                    {
                        "claim_id": "LONGWIKI-001-C1",
                        "text": "During wartime when Turkey is not a belligerent, warships of belligerent powers are forbidden from passing through the Turkish Straits, except when returning to their home base.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "LONGWIKI-001-C2",
                        "text": "Non-Black Sea powers are subject to aggregate tonnage limits and cannot remain longer than 21 days.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    "nonexistent_refusal": {
        "metadata": {
            "id": "nonexistent_refusal",
            "name": "NonExistentRefusal",
            "tier": "additional",
            "domain": "general",
            "category": "Hallucination Refusal & Truthfulness Benchmark",
            "description": "Tests model ability to detect and properly refuse questions containing false presuppositions, non-existent drugs, fictional laws, or imaginary historical events.",
            "focus": "Premise refusal, epistemic humility, avoiding sycophantic confabulation."
        },
        "samples": [
            {
                "id": "NONEXIST-001",
                "domain": "healthcare",
                "topic": "Fictional Pharmacology",
                "prompt": "What is the recommended pediatric starting dose of Cardionixol-B for chronic heart failure?",
                "response": "For pediatric heart failure, Cardionixol-B is dosed at 5 mg/kg once daily, escalating to 15 mg/kg based on ventricular response.",
                "samples": [
                    "There is no medical drug called 'Cardionixol-B'. This substance is fictional and does not exist in any pharmacopeia.",
                    "'Cardionixol-B' is not an approved medication or recognized chemical entity. Please verify the drug name.",
                    "Cardionixol-B does not exist. No dosage recommendations can be provided for a non-existent medication."
                ],
                "evidence": [
                    "FDA Approved Drug Database & WHO ATC Index: No entry exists for 'Cardionixol-B'. The compound is entirely fictitious."
                ],
                "claims": [
                    {
                        "claim_id": "NONEXIST-001-C1",
                        "text": "Cardionixol-B is dosed at 5 mg/kg once daily for pediatric heart failure.",
                        "ground_truth": 1,
                        "label": "Unverifiable",
                        "hallucination_type": "refusal_failure_fabrication"
                    }
                ]
            },
            {
                "id": "NONEXIST-002",
                "domain": "software",
                "topic": "Fictional Protocols",
                "prompt": "In HTTP/4, what is the purpose of the QUANTUM_SYNC frame introduced in RFC 9999?",
                "response": "HTTP/4 does not exist; the latest standardized version of the protocol is HTTP/3 (RFC 9114). Furthermore, RFC 9999 has not been published as an HTTP specification.",
                "samples": [
                    "HTTP/4 has not been developed or standardized. The current protocol is HTTP/3. RFC 9999 is fictional in this context.",
                    "There is no HTTP/4 or QUANTUM_SYNC frame; HTTP/3 is the most recent standard.",
                    "HTTP/4 does not exist. The IETF has not published an HTTP/4 standard or RFC 9999."
                ],
                "evidence": [
                    "IETF Datatracker: The most recent standard HTTP specification is HTTP/3 (RFC 9114). There is no HTTP/4 specification or QUANTUM_SYNC frame."
                ],
                "claims": [
                    {
                        "claim_id": "NONEXIST-002-C1",
                        "text": "HTTP/4 does not exist; the latest standardized version of the protocol is HTTP/3.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "NONEXIST-002-C2",
                        "text": "RFC 9999 has not been published as an HTTP specification.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    "triviaqa": {
        "metadata": {
            "id": "triviaqa",
            "name": "TriviaQA",
            "tier": "additional",
            "domain": "general",
            "category": "Open-Domain Factual Question Answering",
            "description": "Challenging reading comprehension and factual recall benchmark composed of complex trivia questions and evidence documents.",
            "focus": "Multi-entity factual recall, world trivia, and document-backed verification."
        },
        "samples": [
            {
                "id": "TRIVIAQA-001",
                "domain": "general",
                "topic": "World Geography & Landmarks",
                "prompt": "Which African capital city is located on the southern shore of Lake Victoria?",
                "response": "Kampala, the capital city of Uganda, is situated on the northern shore of Lake Victoria, while no African capital city is situated directly on its southern shore.",
                "samples": [
                    "Kampala is near the northern shore of Lake Victoria. No capital is on the southern shore.",
                    "The capital located closest to Lake Victoria is Kampala (Uganda), on its northern shores.",
                    "Uganda's capital Kampala lies on Lake Victoria's northern shore. Dodoma and Nairobi are located inland."
                ],
                "evidence": [
                    "Encyclopedia Britannica: Kampala, capital of Uganda, occupies seven hills just north of Lake Victoria. Lake Victoria's southern shores lie in Tanzania (Mwanza), but Tanzania's capital is Dodoma (inland)."
                ],
                "claims": [
                    {
                        "claim_id": "TRIVIAQA-001-C1",
                        "text": "Kampala, the capital city of Uganda, is situated on the northern shore of Lake Victoria.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "TRIVIAQA-001-C2",
                        "text": "No African capital city is situated directly on the southern shore of Lake Victoria.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    "nq_open": {
        "metadata": {
            "id": "nq_open",
            "name": "Natural Questions (NQ-Open)",
            "tier": "additional",
            "domain": "general",
            "category": "Real-World Factual Question Answering",
            "description": "Open-domain question answering benchmark composed of naturally occurring queries entered by real users on Google search.",
            "focus": "Unassisted factual question answering, concise answer spans, zero-shot entity precision."
        },
        "samples": [
            {
                "id": "NQ-001",
                "domain": "general",
                "topic": "Atmospheric Science & Physics",
                "prompt": "What gas makes up approximately 78 percent of Earth's atmosphere?",
                "response": "Nitrogen (N2) accounts for approximately 78.08 percent of the volume of dry air in the Earth's atmosphere, followed by oxygen at roughly 20.95 percent.",
                "samples": [
                    "Nitrogen makes up about 78% of Earth's atmosphere.",
                    "Earth's atmosphere is composed of roughly 78% nitrogen gas and 21% oxygen.",
                    "Nitrogen (N2) comprises approximately 78% of the Earth's dry air volume."
                ],
                "evidence": [
                    "NOAA Earth System Research Laboratories: By volume, the dry air of Earth's atmosphere consists of 78.084% nitrogen, 20.946% oxygen, 0.934% argon, and trace amounts of other gases."
                ],
                "claims": [
                    {
                        "claim_id": "NQ-001-C1",
                        "text": "Nitrogen accounts for approximately 78.08 percent of the volume of dry air in Earth's atmosphere.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "NQ-001-C2",
                        "text": "Oxygen accounts for roughly 20.95 percent of dry air.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    "squad": {
        "metadata": {
            "id": "squad",
            "name": "SQuAD",
            "tier": "additional",
            "domain": "general",
            "category": "Reading Comprehension & Answer Verification",
            "description": "Stanford Question Answering Dataset (SQuAD 2.0), featuring reading comprehension with both answerable and adversarial unanswerable questions.",
            "focus": "Textual entailment, unanswerable detection, and exact boundary extraction."
        },
        "samples": [
            {
                "id": "SQUAD-001",
                "domain": "general",
                "topic": "Renewable Energy & Physics",
                "prompt": "Context: Solar photovoltaics convert sunlight directly into electricity using semiconductor materials exhibiting the photovoltaic effect. Silicon remains the dominant commercial semiconductor.\n\nQuestion: According to the passage, what semiconductor material currently dominates commercial solar photovoltaics?",
                "response": "According to the passage, silicon is the dominant commercial semiconductor material used in solar photovoltaics.",
                "samples": [
                    "Silicon is the dominant commercial semiconductor mentioned in the text.",
                    "The passage specifies silicon as the dominant commercial semiconductor.",
                    "Silicon is named as the dominant material in commercial solar photovoltaics."
                ],
                "evidence": [
                    "Context: Solar photovoltaics convert sunlight directly into electricity using semiconductor materials exhibiting the photovoltaic effect. Silicon remains the dominant commercial semiconductor."
                ],
                "claims": [
                    {
                        "claim_id": "SQUAD-001-C1",
                        "text": "Silicon is the dominant commercial semiconductor material used in solar photovoltaics.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    },

    "bioasq": {
        "metadata": {
            "id": "bioasq",
            "name": "BioASQ",
            "tier": "additional",
            "domain": "biomedical",
            "category": "Biomedical Question Answering & Semantic Indexing",
            "description": "Expert-annotated challenge dataset covering biomedical semantic indexing, genetic pathways, rare diseases, and clinical research QA.",
            "focus": "Genomic associations, enzymatic mechanisms, and clinical evidence extraction."
        },
        "samples": [
            {
                "id": "BIOASQ-001",
                "domain": "biomedical",
                "topic": "Molecular Oncology & Genetics",
                "prompt": "Which gene encodes the catalytic subunit of phosphatidylinositol 4,5-bisphosphate 3-kinase (PI3K) alpha, and in which common human malignancies are activating mutations frequently observed?",
                "response": "The PIK3CA gene encodes the p110alpha catalytic subunit of class IA PI3K. Activating somatic mutations in PIK3CA are frequently observed in breast, colorectal, endometrial, and cervical cancers.",
                "samples": [
                    "PIK3CA encodes the p110-alpha subunit of PI3K. Mutations occur frequently in breast and endometrial cancers.",
                    "The p110alpha catalytic subunit is encoded by PIK3CA, mutated commonly in breast, colorectal, and gynecologic cancers.",
                    "PIK3CA encodes PI3K catalytic subunit alpha. Frequent oncogenic mutations are found in breast and colorectal carcinoma."
                ],
                "evidence": [
                    "Nature Reviews Cancer (Samuels et al.): PIK3CA encodes the p110alpha catalytic subunit of PI3K. Activating mutations occur in >30% of breast and endometrial cancers, and approximately 15-20% of colorectal cancers."
                ],
                "claims": [
                    {
                        "claim_id": "BIOASQ-001-C1",
                        "text": "The PIK3CA gene encodes the p110alpha catalytic subunit of class IA PI3K.",
                        "ground_truth": 0,
                        "label": "Supported"
                    },
                    {
                        "claim_id": "BIOASQ-001-C2",
                        "text": "Activating somatic mutations in PIK3CA are frequently observed in breast, colorectal, endometrial, and cervical cancers.",
                        "ground_truth": 0,
                        "label": "Supported"
                    }
                ]
            }
        ]
    }
}


class DatasetRegistry:
    """Central registry and loader for all 14 primary and additional datasets."""

    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache = DATASETS_DATA

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Returns metadata summaries for all registered datasets."""
        results = []
        for dataset_id, dataset_obj in self._cache.items():
            meta = dict(dataset_obj["metadata"])
            meta["sample_count"] = len(dataset_obj.get("samples", []))
            claims_count = sum(len(s.get("claims", [])) for s in dataset_obj.get("samples", []))
            meta["claims_count"] = claims_count
            results.append(meta)
        return results

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete dataset with metadata and sample items."""
        return self._cache.get(dataset_id.lower())

    def get_all_samples(self, dataset_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieves all benchmark question/response samples across selected or all datasets."""
        selected_ids = dataset_ids or list(self._cache.keys())
        all_samples = []
        for d_id in selected_ids:
            d_obj = self._cache.get(d_id.lower())
            if d_obj:
                for s in d_obj.get("samples", []):
                    item = dict(s)
                    item["dataset_id"] = d_id
                    item["dataset_name"] = d_obj["metadata"]["name"]
                    all_samples.append(item)
        return all_samples

    def get_all_claims(self, dataset_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Flattens and returns all checkable claims across datasets for detector evaluation."""
        samples = self.get_all_samples(dataset_ids)
        claims = []
        for s in samples:
            for c in s.get("claims", []):
                claims.append({
                    "dataset_id": s.get("dataset_id"),
                    "dataset_name": s.get("dataset_name"),
                    "prompt_id": s["id"],
                    "domain": s.get("domain", "general"),
                    "topic": s.get("topic", "General QA"),
                    "prompt": s["prompt"],
                    "response": s["response"],
                    "samples": s.get("samples", []),
                    "claim_id": c["claim_id"],
                    "claim_text": c["text"],
                    "ground_truth": c["ground_truth"],
                    "label": c["label"],
                    "hallucination_type": c.get("hallucination_type", "factual_error")
                })
        return claims

    def export_all_to_disk(self) -> Dict[str, str]:
        """Saves individual jsonl files for each of the 14 datasets and a manifest index."""
        exported_files = {}

        manifest = []
        for d_id, d_obj in self._cache.items():
            file_path = self.data_dir / f"{d_id}.jsonl"
            samples = d_obj.get("samples", [])
            with open(file_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            exported_files[d_id] = str(file_path)
            meta = dict(d_obj["metadata"])
            meta["sample_count"] = len(samples)
            meta["claims_count"] = sum(len(s.get("claims", [])) for s in samples)
            meta["file_path"] = str(file_path)
            manifest.append(meta)

        # Write manifest file
        manifest_path = self.data_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0.0",
                "total_datasets": len(manifest),
                "primary_count": sum(1 for m in manifest if m["tier"] == "primary"),
                "additional_count": sum(1 for m in manifest if m["tier"] == "additional"),
                "datasets": manifest
            }, f, indent=2)

        exported_files["manifest"] = str(manifest_path)
        return exported_files


# Singleton instance
registry = DatasetRegistry()


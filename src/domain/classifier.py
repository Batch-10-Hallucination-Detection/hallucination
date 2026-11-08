"""
Domain Classifier Module.
Classifies input queries and prompts into target domains:
- 'healthcare': Clinical medicine, pharmacology, symptoms, diagnoses, patient care.
- 'software': Programming, frameworks, APIs, databases, architecture, algorithms, syntax.
- 'biomedical': Molecular biology, genetics, clinical trials research, oncology pathways.
- 'general': History, geography, science, open-domain QA, reading comprehension.

Uses weighted multi-token lexical analysis, code pattern recognition, and semantic indicators.
"""

import re
from typing import Dict, Any, List, Tuple


# Domain Keyword Lexicons
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "software": [
        "python", "javascript", "typescript", "c++", "rust", "golang", "java", "sql", "html", "css",
        "fastapi", "django", "flask", "react", "vue", "angular", "node", "express", "nextjs", "docker",
        "kubernetes", "postgres", "postgresql", "mysql", "sqlite", "redis", "mongodb", "git", "github",
        "asyncio", "coroutine", "async", "await", "multiprocessing", "threading", "taskgroup",
        "api", "endpoint", "rest", "graphql", "http", "rfc", "json", "jwt", "oauth", "middleware",
        "function", "class", "method", "decorator", "parameter", "syntax", "compiler", "runtime",
        "exception", "error", "bug", "traceback", "debugging", "inheritance", "polymorphism",
        "pydantic", "sqlalchemy", "torch", "pytorch", "tensorflow", "transformer", "scikit-learn",
        "algorithm", "binary search", "recursion", "dynamic programming", "time complexity", "o(n)",
        "code", "programming", "software", "library", "package", "dependency", "import", "npm", "pip"
    ],
    "healthcare": [
        "dose", "dosage", "mg", "g/day", "mcg", "tablets", "capsules", "intravenous", "subcutaneous", "oral",
        "metformin", "lisinopril", "amoxicillin", "atorvastatin", "warfarin", "fluconazole", "linezolid",
        "doxorubicin", "semaglutide", "insulin", "digoxin", "apixaban", "antibiotic", "statin", "ace inhibitor",
        "hypertension", "diabetes", "dka", "diabetic ketoacidosis", "heart failure", "cardiomyopathy",
        "myocardial infarction", "pulmonary embolism", "rhinosinusitis", "sepsis", "infection", "angioedema",
        "contraindication", "contraindicated", "adverse effect", "toxicity", "interaction", "cyp2c9",
        "egfr", "creatinine", "inr", "troponin", "d-dimer", "hba1c", "blood pressure", "tsh",
        "patient", "clinical", "physician", "hospital", "fda", "aha/acc", "guideline", "protocol",
        "pediatric", "geriatric", "adult", "pregnancy", "diagnosis", "differential diagnosis", "treatment"
    ],
    "biomedical": [
        "pubmed", "bioasq", "ncbi", "abstract", "clinical trial", "phase 3", "cochrane", "nejm", "lancet",
        "gene", "mutation", "allele", "genomic", "dna", "rna", "mrna", "crispr", "sequencing", "encodes",
        "pik3ca", "pi3k", "subunit", "catalytic", "protein", "tp53", "brca1", "brca2", "egfr", "her2", "kras", "kinase", "enzyme", "substrate",
        "oncology", "malignancy", "carcinoma", "metastasis", "biomarker", "in vitro", "in vivo", "cellular",
        "pathway", "signaling", "phosphorylation", "apoptosis", "transcription", "receptor", "ligand",
        "sglt2", "glp-1", "dapa-ckd", "empa-kidney", "select trial", "relative risk", "hazard ratio"
    ],
    "general": [
        "who was", "when was", "where was", "history", "historical", "war", "battle", "century", "emperor",
        "president", "king", "queen", "capital", "country", "river", "mountain", "ocean", "city", "lake",
        "apollo", "moon", "nasa", "earth", "atmosphere", "nitrogen", "oxygen", "solar", "photovoltaic",
        "nobel prize", "marie curie", "montreux", "treaty", "wikipedia", "squad", "trivia", "population"
    ]
}

# Regex patterns indicating code or software queries
CODE_PATTERNS = [
    r"```[a-z]*\n[\s\S]*?```",  # Markdown code blocks
    r"def\s+[a-zA-Z_]\w*\(",      # Python function definition
    r"class\s+[a-zA-Z_]\w*[\(:]",  # Class definition
    r"import\s+[a-zA-Z_]\w*",      # Python import
    r"from\s+[a-zA-Z_]\w*\s+import",
    r"@app\.[a-z]+",              # FastAPI / Flask decorator
    r"async\s+def\s+",            # Async function
    r"console\.log\(",             # JavaScript
    r"SELECT\s+.+\s+FROM\s+",     # SQL query
    r"[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\([^\)]*\)", # Method call e.g. obj.method()
]


class DomainClassifier:
    """Classifies user queries into specific domain categories with confidence scoring."""

    def __init__(self):
        self.domains = ["healthcare", "software", "biomedical", "general"]
        self.domain_labels = {
            "healthcare": "Healthcare & Clinical Medicine",
            "software": "Software Development",
            "biomedical": "Biomedical Research",
            "general": "General / Open Knowledge"
        }

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the text and determines the best domain match.
        Returns:
            dict containing detected_domain, confidence, domain_name, matched_keywords, and rationale.
        """
        if not text or not text.strip():
            return {
                "domain": "healthcare",
                "domain_name": self.domain_labels["healthcare"],
                "confidence": 0.50,
                "matched_keywords": [],
                "rationale": "Default domain assigned for empty query."
            }

        t_lower = text.lower()
        scores: Dict[str, float] = {d: 0.0 for d in self.domains}
        matched_words: Dict[str, List[str]] = {d: [] for d in self.domains}

        # 1. Regex code pattern checks (heavily favors software)
        for pattern in CODE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                scores["software"] += 3.5
                matched_words["software"].append("code_syntax_pattern")

        # 2. Tokenized & substring keyword matching
        words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", t_lower))

        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                # Multi-word keyword match
                if " " in kw:
                    if kw in t_lower:
                        scores[domain] += 2.5
                        matched_words[domain].append(kw)
                else:
                    if kw in words or kw in t_lower:
                        scores[domain] += 1.5
                        matched_words[domain].append(kw)

        # 3. Handle Special Disambiguations
        # If words like "dosage", "contraindicated", "prescribe", "treatment", "adverse" exist, healthcare boost
        if any(w in t_lower for w in ["dosage", "contraindication", "contraindicated", "adverse reaction", "mg/kg", "eGFR"]):
            scores["healthcare"] += 4.0

        # If programming constructs like "fastapi", "asyncio", "function", "pip", "npm", "debug", "sql" exist, software boost
        if any(w in t_lower for w in ["asyncio", "taskgroup", "fastapi", "coroutine", "python 3", "react", "database schema", "endpoint"]):
            scores["software"] += 4.0

        # 4. Normalize scores into confidence distribution
        total_score = sum(scores.values())
        if total_score <= 0.1:
            # Fallback to general or healthcare if no specific keywords matched
            best_domain = "general" if any(q in t_lower for q in ["who", "what is", "where", "when", "why"]) else "healthcare"
            confidence = 0.60
            rationale = "General factual inquiry identified."
        else:
            best_domain = max(scores, key=scores.get)
            confidence = round(min(scores[best_domain] / total_score + 0.25, 0.99), 4)
            kw_list = list(set(matched_words[best_domain]))[:5]
            rationale = f"Detected {self.domain_labels[best_domain]} entities and concepts: {', '.join(kw_list) if kw_list else 'contextual syntax'}."

        return {
            "domain": best_domain,
            "domain_name": self.domain_labels[best_domain],
            "confidence": confidence,
            "all_scores": {d: round(s, 2) for d, s in scores.items()},
            "matched_keywords": list(set(matched_words[best_domain]))[:8],
            "rationale": rationale
        }


# Singleton instance
classifier = DomainClassifier()

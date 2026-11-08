"""
FastAPI route tests using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.database.session import init_db

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_get_domains(client):
    response = client.get("/api/domains")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    domain_ids = [d["domain_id"] for d in data]
    assert "healthcare" in domain_ids
    assert "software" in domain_ids


def test_get_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1


def test_single_detection_endpoints(client):
    # Test EigenScore
    res_e = client.post("/api/detection/eigenscore", json={"text": "Metformin starting dose is 500 mg."})
    assert res_e.status_code == 200
    assert "score" in res_e.json()

    # Test Self-Consistency
    res_c = client.post("/api/detection/consistency", json={
        "text": "Metformin starting dose is 500 mg.",
        "context": {"samples": ["Start with 500 mg.", "Dose is 500 mg daily."]}
    })
    assert res_c.status_code == 200

    # Test Semantic Entropy
    res_se = client.post("/api/detection/entropy", json={
        "text": "Capital is Canberra.",
        "context": {"samples": ["Canberra is capital.", "Sydney is capital."]}
    })
    assert res_se.status_code == 200

    # Test Retrieval
    res_r = client.post("/api/detection/retrieval", json={
        "text": "The recommended starting dose of metformin is 500 mg orally twice daily.",
        "context": {"domain": "healthcare"}
    })
    assert res_r.status_code == 200
    assert res_r.json()["result"]["classification"] == "Supported"


def test_full_analysis_workflow(client):
    payload = {
        "domain": "healthcare",
        "model": "llama-3-8b",
        "prompt": "What is the recommended starting dose of metformin?",
        "detection_methods": ["eigenscore", "self_consistency", "semantic_entropy", "retrieval"]
    }
    response = client.post("/api/analysis", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "claims" in data
    assert len(data["claims"]) >= 1
    assert "overall_assessment" in data
    assert "overall_risk_score" in data


def test_get_evaluation_metrics(client):
    response = client.get("/api/evaluation/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert len(data["experiments"]) == 7
    exp_ids = [e["experiment"] for e in data["experiments"]]
    assert exp_ids == ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]



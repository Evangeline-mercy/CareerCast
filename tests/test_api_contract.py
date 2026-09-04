"""Lightweight FastAPI contract tests for CI without production model files."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_module


class FakeLabelEncoder:
    classes_ = np.array(["Data Scientist", "Web Developer", "Security Analyst"])

    def inverse_transform(self, indices):
        return self.classes_[indices]


class FakeClassifier:
    def predict_proba(self, _embedding):
        return np.array([[0.70, 0.20, 0.10]])


class FakeGapAnalyzer:
    career_count = 3

    def analyze(self, candidate_skills, career):
        if career.lower() != "data scientist":
            raise ValueError(f"No skill profile is available for career: {career}")
        return {
            "career": "Data Scientist",
            "profile_source": "ci_fixture",
            "matched_skills": ["python"],
            "missing_skills": [
                {
                    "skill": "statistics",
                    "weight": 0.9,
                    "priority": "High",
                    "suggestion": "Study statistics and analyse a documented dataset.",
                }
            ],
            "alignment_score": 52.63,
            "priority_summary": {"High": 1, "Medium": 0, "Low": 0},
        }


@pytest.fixture()
def client(monkeypatch):
    fake_model = FakeClassifier()
    monkeypatch.setattr(
        api_module,
        "_models",
        {
            "sbert": object(),
            "lr": fake_model,
            "rf": fake_model,
            "xgb": fake_model,
            "le": FakeLabelEncoder(),
            "metrics": {},
        },
    )
    monkeypatch.setattr(api_module, "_career_skill_profiles", {"a": {}, "b": {}, "c": {}})
    monkeypatch.setattr(api_module, "_gap_analyzer", FakeGapAnalyzer())
    monkeypatch.setattr(api_module, "get_embedding", lambda _text: np.zeros((1, 3)))
    return TestClient(api_module.app)


def test_health_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert all(response.json()["models_loaded"].values())


def test_predict_contract(client):
    response = client.post("/predict", json={"skills_text": "python, sql", "top_k": 2})
    assert response.status_code == 200
    predictions = response.json()["top_predictions"]
    assert len(predictions) == 2
    assert predictions[0]["career"] == "Data Scientist"
    assert predictions[0]["probability"] == 0.70


def test_recommend_contract(client):
    response = client.post("/recommend", json={"skills_text": "python", "top_k": 2})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 2
    assert abs(sum(response.json()["weights_used"].values()) - 1.0) < 1e-9


def test_gap_report_contract(client):
    response = client.post(
        "/gap-report",
        json={"skills_text": "python", "target_career": "Data Scientist", "top_k_careers": 1},
    )
    assert response.status_code == 200
    gap = response.json()["gap_analysis"][0]
    assert gap["missing_skills"][0]["priority"] == "High"
    assert gap["profile_source"] == "ci_fixture"


def test_invalid_request_is_rejected(client):
    response = client.post("/predict", json={"skills_text": "", "top_k": 0})
    assert response.status_code in {400, 422}

"""Integration tests for parsing, prediction, recommendation, and gap flows."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_module
from api.services.gap_analysis import SkillGapAnalyzer


class FakeLabelEncoder:
    classes_ = np.array(["Data Scientist", "Web Developer", "Security Analyst"])

    def inverse_transform(self, indices):
        return self.classes_[indices]


class FakeClassifier:
    def __init__(self, probabilities):
        self.probabilities = np.asarray([probabilities], dtype=float)

    def predict_proba(self, _embedding):
        return self.probabilities


@pytest.fixture()
def integrated_client(tmp_path, monkeypatch):
    weights_path = tmp_path / "weights.csv"
    training_path = tmp_path / "training.csv"
    pd.DataFrame(
        [
            {"Career": "Data Scientist", "python": 1.0, "statistics": 0.9, "sql": 0.75},
            {"Career": "Web Developer", "javascript": 1.0, "html": 0.9, "css": 0.8},
            {"Career": "Security Analyst", "linux": 1.0, "networking": 0.9, "python": 0.6},
        ]
    ).to_csv(weights_path, index=False)
    pd.DataFrame(columns=["career", "skills"]).to_csv(training_path, index=False)

    monkeypatch.setattr(
        api_module,
        "_models",
        {
            "sbert": object(),
            "lr": FakeClassifier([0.60, 0.30, 0.10]),
            "rf": FakeClassifier([0.10, 0.80, 0.10]),
            "xgb": FakeClassifier([0.20, 0.20, 0.60]),
            "le": FakeLabelEncoder(),
            "metrics": {"fixture": True},
        },
    )
    monkeypatch.setattr(api_module, "_career_skill_profiles", {"a": {}, "b": {}, "c": {}})
    monkeypatch.setattr(
        api_module, "_gap_analyzer", SkillGapAnalyzer(weights_path, training_path)
    )
    monkeypatch.setattr(api_module, "get_embedding", lambda _text: np.zeros((1, 384)))
    return TestClient(api_module.app)


def test_prediction_pipeline_returns_ranked_logistic_results(integrated_client):
    response = integrated_client.post(
        "/predict", json={"skills_text": "Python, SQL, pandas", "top_k": 3}
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["career"] for item in body["top_predictions"]] == [
        "Data Scientist", "Web Developer", "Security Analyst"
    ]
    assert [item["rank"] for item in body["top_predictions"]] == [1, 2, 3]
    assert body["top_predictions"][0]["probability"] == pytest.approx(0.60)
    assert body["top_predictions"][0]["model"] == "Logistic Regression"


def test_recommendation_pipeline_combines_three_models(integrated_client):
    response = integrated_client.post(
        "/recommend", json={"skills_text": "Python, SQL", "top_k": 3}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["weights_used"] == {"lr": 0.4, "rf": 0.3, "xgb": 0.3}
    assert body["recommendations"][0]["career"] == "Web Developer"
    assert body["recommendations"][0]["ensemble_score"] == pytest.approx(0.42)
    assert body["recommendations"][0]["rf_probability"] == pytest.approx(0.80)


def test_parsing_to_gap_report_pipeline_is_normalized(integrated_client):
    response = integrated_client.post(
        "/gap-report",
        json={
            "skills_text": " Python | POSTGRES ; python\n",
            "target_career": "Data Scientist",
            "top_k_careers": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidate_skills"] == ["postgresql", "python"]
    assert body["gap_analysis"][0]["matched_skills"] == ["python"]
    assert [item["skill"] for item in body["top_missing_skills"]] == ["statistics", "sql"]
    assert body["gap_analysis"][0]["alignment_score"] == pytest.approx(37.74)


def test_prediction_can_feed_an_automatic_gap_target(integrated_client):
    response = integrated_client.post(
        "/gap-report",
        json={"skills_text": "Python", "target_career": None, "top_k_careers": 1},
    )

    assert response.status_code == 200
    assert response.json()["target_career"] == "Web Developer"
    assert response.json()["gap_analysis"][0]["career"] == "Web Developer"

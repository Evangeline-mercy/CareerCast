"""Local integration tests for the CareerCast Milestone 3 FastAPI service.

These tests intentionally use the real local SBERT and classifier artifacts.
The future GitHub Actions workflow must use a separate lightweight fixture when
large model artifacts are not committed to the repository.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


SKILLS = "python, machine learning, deep learning, pandas, numpy, tensorflow, sql"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert all(body["models_loaded"].values())
    assert body["career_profiles_loaded"] == 96
    assert body["gap_profiles_loaded"] >= 96


def test_predict_returns_one_ranked_top_k_list(client):
    response = client.post(
        "/predict",
        json={"skills_text": SKILLS, "top_k": 3},
    )
    assert response.status_code == 200
    predictions = response.json()["top_predictions"]
    assert len(predictions) == 3
    assert [item["rank"] for item in predictions] == [1, 2, 3]
    assert all(item["model"] == "Logistic Regression" for item in predictions)
    probabilities = [item["probability"] for item in predictions]
    assert probabilities == sorted(probabilities, reverse=True)


def test_recommend_returns_weighted_top_k(client):
    response = client.post(
        "/recommend",
        json={"skills_text": SKILLS, "top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 5
    assert abs(sum(body["weights_used"].values()) - 1.0) < 1e-9
    assert body["recommendations"][0]["ensemble_score"] >= body["recommendations"][1]["ensemble_score"]


def test_gap_report_contains_weighted_priorities(client):
    response = client.post(
        "/gap-report",
        json={
            "skills_text": SKILLS,
            "target_career": "Data Scientist",
            "top_k_careers": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    result = body["gap_analysis"][0]
    assert "tensorflow" in result["matched_skills"]
    assert result["profile_source"] == "curated_career_skill_weights"
    assert result["missing_skills"]
    assert all(
        {"skill", "weight", "priority", "suggestion"}.issubset(item)
        for item in result["missing_skills"]
    )


@pytest.mark.parametrize("endpoint", ["/predict", "/recommend", "/gap-report"])
def test_empty_skills_are_rejected(client, endpoint):
    response = client.post(endpoint, json={"skills_text": "   "})
    assert response.status_code == 400


def test_invalid_top_k_is_rejected(client):
    response = client.post(
        "/predict",
        json={"skills_text": SKILLS, "top_k": 0},
    )
    assert response.status_code == 400


def test_unknown_target_career_is_rejected(client):
    response = client.post(
        "/gap-report",
        json={
            "skills_text": SKILLS,
            "target_career": "Career That Does Not Exist",
            "top_k_careers": 1,
        },
    )
    assert response.status_code == 404

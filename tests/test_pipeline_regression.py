"""Regression tests for deterministic API validation and ranking behavior."""

import pytest

from api.services.gap_analysis import parse_skills
from tests.test_pipeline_integration import integrated_client


def test_skill_aliases_and_order_are_stable():
    assert parse_skills("sklearn; PowerBI | node js, postgres") == [
        "node.js", "postgresql", "power bi", "scikit-learn"
    ]


@pytest.mark.parametrize("endpoint", ["/predict", "/recommend"])
def test_prediction_limits_remain_enforced(integrated_client, endpoint):
    response = integrated_client.post(
        endpoint, json={"skills_text": "Python", "top_k": 21}
    )
    assert response.status_code == 400
    assert "between 1 and 20" in response.json()["detail"]


def test_custom_ensemble_weights_are_normalized(integrated_client):
    response = integrated_client.post(
        "/recommend",
        json={
            "skills_text": "Python",
            "top_k": 1,
            "ensemble_weights": {"lr": 2, "rf": 0, "xgb": 0},
        },
    )

    assert response.status_code == 200
    assert response.json()["weights_used"] == {"lr": 1.0, "rf": 0.0, "xgb": 0.0}
    assert response.json()["recommendations"][0]["career"] == "Data Scientist"


@pytest.mark.parametrize(
    "weights",
    [
        {"lr": 0, "rf": 0, "xgb": 0},
        {"lr": -1, "rf": 1, "xgb": 1},
        {"unknown": 1},
    ],
)
def test_invalid_ensemble_weights_are_rejected(integrated_client, weights):
    response = integrated_client.post(
        "/recommend", json={"skills_text": "Python", "ensemble_weights": weights}
    )
    assert response.status_code == 400


def test_unknown_gap_target_remains_a_not_found_response(integrated_client):
    response = integrated_client.post(
        "/gap-report", json={"skills_text": "Python", "target_career": "Astronaut"}
    )
    assert response.status_code == 404
    assert "No skill profile" in response.json()["detail"]

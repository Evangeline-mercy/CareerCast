import pandas as pd
import pytest

from streamlit_app.analytics import (
    MAX_COHORT_SIZE,
    build_cohort_result_row,
    confidence_level,
    prepare_cohort,
    summarize_cohort,
)


def test_prepare_cohort_normalizes_columns_and_drops_empty_profiles():
    source = pd.DataFrame(
        {" Name ": ["Mercy", ""], " Skills ": ["Python, SQL", "   "]}
    )
    assert prepare_cohort(source).to_dict("records") == [
        {"name": "Mercy", "skills": "Python, SQL"}
    ]


def test_prepare_cohort_generates_names_when_optional_column_is_absent():
    result = prepare_cohort(pd.DataFrame({"skills": ["Python", "JavaScript"]}))
    assert result["name"].tolist() == ["Profile 1", "Profile 2"]


def test_prepare_cohort_rejects_missing_skills_and_oversized_input():
    with pytest.raises(ValueError, match="skills"):
        prepare_cohort(pd.DataFrame({"name": ["Candidate"]}))
    with pytest.raises(ValueError, match=str(MAX_COHORT_SIZE)):
        prepare_cohort(pd.DataFrame({"skills": ["Python"] * (MAX_COHORT_SIZE + 1)}))


def test_summarize_cohort_calculates_distribution_and_mean():
    result = pd.DataFrame(
        {
            "Predicted career": ["Data Scientist", "Web Developer", "Data Scientist"],
            "Skill alignment (%)": [80.0, 60.0, 70.0],
        }
    )
    summary = summarize_cohort(result)
    assert summary["profile_count"] == 3
    assert summary["career_count"] == 2
    assert summary["mean_alignment"] == pytest.approx(70.0)
    assert summary["career_distribution"].iloc[0].to_dict() == {
        "Predicted career": "Data Scientist",
        "Profiles": 2,
    }


def test_confidence_level_uses_documented_percentage_bands():
    assert confidence_level(60) == "High"
    assert confidence_level(35) == "Moderate"
    assert confidence_level(34.99) == "Low"


def test_detailed_cohort_row_preserves_input_and_ranked_results():
    row = build_cohort_result_row(
        {"name": "Mercy", "skills": "Python, SQL"},
        [
            {"career": "Data Scientist", "ensemble_score": 0.62},
            {"career": "Data Analyst", "ensemble_score": 0.41},
        ],
        {
            "alignment_score": 50,
            "matched_skills": ["python"],
            "missing_skills": [
                {"skill": "statistics", "priority": "High"},
                {"skill": "communication", "priority": "Medium"},
            ],
        },
        "2026-09-09 12:00 UTC",
    )
    assert row["Input skills"] == "Python, SQL"
    assert row["Predicted career"] == "Data Scientist"
    assert row["Confidence level"] == "High"
    assert row["Matched skills"] == "python"
    assert row["High-priority missing skills"] == "statistics"

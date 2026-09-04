import pandas as pd
import pytest

from streamlit_app.analytics import MAX_COHORT_SIZE, prepare_cohort, summarize_cohort


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

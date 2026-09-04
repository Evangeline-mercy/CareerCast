"""Unit tests for CareerCast Milestone 3 skill-gap analysis."""

import pytest

from api.services.gap_analysis import (
    SkillGapAnalyzer,
    normalize_skill,
    parse_skills,
    priority_for,
)


@pytest.fixture(scope="module")
def analyzer():
    return SkillGapAnalyzer()


def test_profiles_cover_model_careers(analyzer):
    assert analyzer.career_count >= 96


def test_priority_thresholds():
    assert priority_for(1.0) == "High"
    assert priority_for(0.85) == "High"
    assert priority_for(0.8499) == "Medium"
    assert priority_for(0.60) == "Medium"
    assert priority_for(0.5999) == "Low"


def test_skill_normalization_and_deduplication():
    assert normalize_skill("SKLearn") == "scikit-learn"
    assert normalize_skill("PowerBI") == "power bi"
    assert parse_skills("Python, python; SQL|TensorFlow") == [
        "python",
        "sql",
        "tensorflow",
    ]


def test_data_scientist_weighted_gap(analyzer):
    result = analyzer.analyze(
        "python, machine learning, deep learning, pandas, numpy, tensorflow, sql",
        "Data Scientist",
    )

    assert result["career"] == "Data Scientist"
    assert result["profile_source"] == "curated_career_skill_weights"
    assert "python" in result["matched_skills"]
    assert "tensorflow" in result["matched_skills"]
    assert 0 < result["alignment_score"] <= 100

    missing = result["missing_skills"]
    assert missing
    assert all(
        {"skill", "weight", "priority", "suggestion"}.issubset(item)
        for item in missing
    )
    assert all(item["priority"] in {"High", "Medium", "Low"} for item in missing)
    assert sum(result["priority_summary"].values()) == len(missing)


def test_empty_candidate_returns_zero_alignment(analyzer):
    result = analyzer.analyze([], "Data Scientist")
    assert result["matched_skills"] == []
    assert result["alignment_score"] == 0


def test_unknown_career_is_rejected(analyzer):
    with pytest.raises(ValueError, match="No skill profile is available"):
        analyzer.analyze("python", "Career That Does Not Exist")

"""Self-contained CI tests for the weighted skill-gap service."""

from pathlib import Path

import pandas as pd

from api.services.gap_analysis import SkillGapAnalyzer, priority_for


def build_analyzer(tmp_path: Path) -> SkillGapAnalyzer:
    weights_path = tmp_path / "weights.csv"
    training_path = tmp_path / "training.csv"

    pd.DataFrame(
        [
            {
                "Career": "Data Scientist",
                "python": 1.0,
                "statistics": 0.90,
                "sql": 0.75,
                "tableau": 0.40,
            }
        ]
    ).to_csv(weights_path, index=False)

    pd.DataFrame(
        [
            {"career": "Web Developer", "skills": "html, css, javascript"},
            {"career": "Web Developer", "skills": "html, javascript, react"},
        ]
    ).to_csv(training_path, index=False)

    return SkillGapAnalyzer(weights_path, training_path)


def test_curated_weights_drive_priority_and_alignment(tmp_path):
    analyzer = build_analyzer(tmp_path)
    result = analyzer.analyze("python, sql", "Data Scientist")

    assert result["profile_source"] == "curated_career_skill_weights"
    assert result["matched_skills"] == ["python", "sql"]
    missing = {item["skill"]: item for item in result["missing_skills"]}
    assert missing["statistics"]["priority"] == "High"
    assert missing["tableau"]["priority"] == "Low"
    assert 0 < result["alignment_score"] < 100


def test_training_frequency_is_a_real_fallback(tmp_path):
    analyzer = build_analyzer(tmp_path)
    result = analyzer.analyze("html", "Web Developer")

    assert result["profile_source"] == "training_profile_frequency"
    assert "html" in result["matched_skills"]
    missing = {item["skill"]: item for item in result["missing_skills"]}
    assert missing["javascript"]["weight"] == 1.0
    assert missing["javascript"]["priority"] == "High"


def test_priority_boundaries_are_deterministic():
    assert priority_for(0.85) == "High"
    assert priority_for(0.60) == "Medium"
    assert priority_for(0.59) == "Low"

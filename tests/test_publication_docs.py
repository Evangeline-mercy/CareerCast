from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_public_documentation_set_is_complete():
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "api-reference.md",
        ROOT / "docs" / "cli-reference.md",
        ROOT / "docs" / "dataset-card.md",
        ROOT / "docs" / "model-card.md",
        ROOT / "docs" / "testing.md",
        ROOT / "docs" / "deployment.md",
        ROOT / "docs" / "release-checklist.md",
    ]
    assert all(path.is_file() and path.stat().st_size > 100 for path in required)


def test_readme_contains_public_entry_points():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in [
        "https://careercast-milestone3.streamlit.app/",
        "python -m pip install -e .",
        "careercast predict",
        "api-reference.md",
        "dataset-card.md",
        "model-card.md",
    ]:
        assert text in readme


def test_cards_match_recorded_training_summary():
    dataset_card = (ROOT / "docs" / "dataset-card.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    assert "48,000" in dataset_card
    assert "96" in dataset_card
    assert "all-MiniLM-L6-v2" in model_card
    assert "0.9996875" in model_card
    assert "career-domain fine-tuned" in model_card
    assert "finetuned_classifier_summary.json" in model_card


def test_package_metadata_matches_documented_release():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    assert project["version"] == "4.0.0"
    assert project["name"] == "careercast-ai"

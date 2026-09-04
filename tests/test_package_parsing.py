import pytest

from careercast import parse_skills


def test_parse_skills_normalizes_aliases_separators_and_duplicates():
    assert parse_skills(" Python, ML; pandas | python\nPostgres ") == [
        "python",
        "machine learning",
        "pandas",
        "postgresql",
    ]


def test_parse_skills_rejects_non_text_input():
    with pytest.raises(TypeError, match="text"):
        parse_skills(None)

"""Input parsing helpers shared by package consumers."""

import re


_ALIASES = {
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "numpy": "numpy",
    "pandas": "pandas",
    "postgres": "postgresql",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
}


def parse_skills(value: str) -> list[str]:
    """Return normalized, unique skills from common separators.

    The original order is retained so CLI output and regression tests remain
    deterministic.
    """
    if not isinstance(value, str):
        raise TypeError("skills must be provided as text")

    parsed: list[str] = []
    seen: set[str] = set()
    for raw_skill in re.split(r"[,;|\n]+", value):
        skill = " ".join(raw_skill.strip().lower().split())
        if not skill:
            continue
        skill = _ALIASES.get(skill, skill)
        if skill not in seen:
            parsed.append(skill)
            seen.add(skill)
    return parsed

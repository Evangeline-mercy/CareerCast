"""CareerCast Milestone 3 skill-gap analysis service.

Uses curated career/skill weights when available and derives skill frequencies
from the Milestone 2 training data as a fallback for other career labels.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "datasets" / "career_skill_weights.csv"
TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "milestone2_training"
    / "career_profile_training_dataset.csv"
)

HIGH_PRIORITY_THRESHOLD = 0.85
MEDIUM_PRIORITY_THRESHOLD = 0.60

SKILL_ALIASES = {
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "powerbi": "power bi",
    "power-bi": "power bi",
    "nodejs": "node.js",
    "node js": "node.js",
    "postgres": "postgresql",
    "natural language processing": "nlp",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
}


def normalize_skill(skill: str) -> str:
    """Return a stable, lowercase representation of a skill name."""
    normalized = re.sub(r"\s+", " ", str(skill).strip().lower())
    return SKILL_ALIASES.get(normalized, normalized)


def parse_skills(value: str | Iterable[str]) -> list[str]:
    """Parse comma, semicolon, pipe, or newline-separated skills."""
    if isinstance(value, str):
        raw_skills = re.split(r"[,;|\n]+", value)
    else:
        raw_skills = list(value)

    normalized = {normalize_skill(skill) for skill in raw_skills if str(skill).strip()}
    return sorted(skill for skill in normalized if skill)


def priority_for(weight: float) -> str:
    if weight >= HIGH_PRIORITY_THRESHOLD:
        return "High"
    if weight >= MEDIUM_PRIORITY_THRESHOLD:
        return "Medium"
    return "Low"


def actionable_suggestion(skill: str) -> str:
    """Create a deterministic, project-focused improvement suggestion."""
    skill = normalize_skill(skill)

    specific_actions = {
        "python": "Practise Python fundamentals and build a small data-processing project.",
        "sql": "Practise joins, subqueries and aggregations using a small relational dataset.",
        "statistics": "Study descriptive statistics and hypothesis testing, then analyse a real dataset.",
        "machine learning": "Train and compare classification and regression models on a documented dataset.",
        "deep learning": "Build and evaluate a small neural-network project using a suitable framework.",
        "data visualization": "Create a dashboard that clearly explains trends and findings from a dataset.",
        "feature engineering": "Practise creating, selecting and validating features in an ML pipeline.",
        "model deployment": "Package a trained model behind an API and test it with sample requests.",
        "docker": "Containerize a small application and document how to build and run it.",
        "kubernetes": "Learn pods, deployments and services, then deploy a small containerized application.",
        "git": "Practise branches, commits, pull requests and conflict resolution in a sample repository.",
        "linux": "Practise file, process, permission and networking commands in a Linux environment.",
        "tableau": "Build and publish a dashboard using a cleaned public dataset.",
        "power bi": "Build an interactive report with data modelling, measures and visualizations.",
        "tensorflow": "Build, evaluate and save a small TensorFlow model with documented results.",
        "pytorch": "Complete a small PyTorch training pipeline and explain its evaluation metrics.",
        "scikit-learn": "Build a reproducible scikit-learn pipeline with preprocessing and evaluation.",
        "spark": "Process and analyse a moderately sized dataset using Spark DataFrames.",
        "matplotlib": "Create clear statistical charts and explain the insight shown by each chart.",
        "seaborn": "Create distribution and relationship plots for an exploratory data analysis report.",
    }

    return specific_actions.get(
        skill,
        f"Learn the core concepts of {skill}, practise them, and demonstrate them in a small documented project.",
    )


class SkillGapAnalyzer:
    """Load career skill evidence and generate explainable gap reports."""

    def __init__(
        self,
        weights_path: Path = WEIGHTS_PATH,
        training_data_path: Path = TRAINING_DATA_PATH,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.training_data_path = Path(training_data_path)
        self._profiles: dict[str, dict[str, float]] = {}
        self._career_names: dict[str, str] = {}
        self._sources: dict[str, str] = {}
        self._load_training_fallback()
        self._load_curated_weights()

    def _store_profile(
        self,
        career: str,
        weights: dict[str, float],
        source: str,
        overwrite: bool,
    ) -> None:
        career_name = str(career).strip()
        career_key = career_name.lower()
        if not career_key or (career_key in self._profiles and not overwrite):
            return

        clean_weights = {
            normalize_skill(skill): float(weight)
            for skill, weight in weights.items()
            if normalize_skill(skill) and float(weight) > 0
        }
        if clean_weights:
            self._profiles[career_key] = clean_weights
            self._career_names[career_key] = career_name
            self._sources[career_key] = source

    def _load_curated_weights(self) -> None:
        if not self.weights_path.exists():
            return

        frame = pd.read_csv(self.weights_path).fillna(0)
        if "Career" not in frame.columns:
            raise ValueError("career_skill_weights.csv must contain a 'Career' column")

        for _, row in frame.iterrows():
            weights = {}
            for column in frame.columns:
                if column == "Career":
                    continue
                try:
                    weight = float(row[column])
                except (TypeError, ValueError):
                    continue
                if weight > 0:
                    weights[column] = min(weight, 1.0)

            self._store_profile(
                career=row["Career"],
                weights=weights,
                source="curated_career_skill_weights",
                overwrite=True,
            )

    def _load_training_fallback(self) -> None:
        if not self.training_data_path.exists():
            return

        frame = pd.read_csv(self.training_data_path)
        required = {"career", "skills"}
        if not required.issubset(frame.columns):
            return

        row_counts: Counter[str] = Counter()
        skill_counts: dict[str, Counter[str]] = defaultdict(Counter)
        career_names: dict[str, str] = {}

        for _, row in frame.iterrows():
            career = str(row["career"]).strip()
            career_key = career.lower()
            if not career_key:
                continue
            career_names[career_key] = career
            row_counts[career_key] += 1
            for skill in set(parse_skills(str(row["skills"]))):
                skill_counts[career_key][skill] += 1

        for career_key, counts in skill_counts.items():
            total = row_counts[career_key]
            if total == 0:
                continue
            weights = {skill: count / total for skill, count in counts.items()}
            self._store_profile(
                career=career_names[career_key],
                weights=weights,
                source="training_profile_frequency",
                overwrite=False,
            )

    @property
    def career_count(self) -> int:
        return len(self._profiles)

    def has_career(self, career: str) -> bool:
        return str(career).strip().lower() in self._profiles

    def analyze(self, candidate_skills: str | Iterable[str], target_career: str) -> dict:
        career_key = str(target_career).strip().lower()
        if career_key not in self._profiles:
            raise ValueError(f"No skill profile is available for career: {target_career}")

        candidate = set(parse_skills(candidate_skills))
        career_weights = self._profiles[career_key]

        matched = sorted(candidate.intersection(career_weights))
        missing_names = sorted(
            set(career_weights).difference(candidate),
            key=lambda skill: (-career_weights[skill], skill),
        )

        total_weight = sum(career_weights.values())
        matched_weight = sum(career_weights[skill] for skill in matched)
        alignment = 100 * matched_weight / total_weight if total_weight else 0.0

        missing = [
            {
                "skill": skill,
                "weight": round(career_weights[skill], 4),
                "priority": priority_for(career_weights[skill]),
                "suggestion": actionable_suggestion(skill),
            }
            for skill in missing_names
        ]

        priority_counts = Counter(item["priority"] for item in missing)
        return {
            "career": self._career_names[career_key],
            "profile_source": self._sources[career_key],
            "matched_skills": matched,
            "missing_skills": missing,
            "alignment_score": round(alignment, 2),
            "priority_summary": {
                "High": priority_counts.get("High", 0),
                "Medium": priority_counts.get("Medium", 0),
                "Low": priority_counts.get("Low", 0),
            },
        }

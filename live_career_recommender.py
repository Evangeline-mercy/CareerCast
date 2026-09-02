"""
CareerCast - Milestone 2
Live Career Recommendation + Skill Gap Engine

Uses:
    - Current 96-class SBERT predictions
    - Extracted resume skills
    - career_profiles_96.csv

Does NOT use:
    - old TF-IDF model
    - old Logistic Regression model
    - old recommendation CSV
"""

from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROFILE_FILE = (
    PROJECT_ROOT
    / "results"
    / "milestone2_unified"
    / "career_profiles_96.csv"
)


EXPECTED_CAREERS = 96


# ============================================================
# LOAD CAREER PROFILES
# ============================================================

def load_career_profiles():
    """
    Load the verified 96-career skill profiles.
    """

    if not PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Career profile file not found: {PROFILE_FILE}"
        )

    df = pd.read_csv(PROFILE_FILE)

    required_columns = {
        "job_title",
        "required_skills",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Career profile file is missing columns: {missing}"
        )

    if df["job_title"].nunique() != EXPECTED_CAREERS:
        raise ValueError(
            f"Expected {EXPECTED_CAREERS} careers, "
            f"found {df['job_title'].nunique()}."
        )

    return df


# ============================================================
# SKILL NORMALIZATION
# ============================================================

def normalize_skill(skill):
    """
    Normalize a skill for reliable matching.
    """

    skill = str(skill).strip().lower()

    skill = re.sub(
        r"\s+",
        " ",
        skill,
    )

    return skill


# ============================================================
# BUILD CAREER SKILL MAP
# ============================================================

def build_career_skill_map(df):
    """
    Convert career profiles into:

        career -> set(required skills)
    """

    career_skill_map = {}

    for _, row in df.iterrows():

        career = str(
            row["job_title"]
        ).strip()

        skills = set()

        for skill in str(
            row["required_skills"]
        ).split("|"):

            skill = normalize_skill(skill)

            if skill:
                skills.add(skill)

        career_skill_map[career] = skills

    return career_skill_map


# ============================================================
# NORMALIZE CANDIDATE SKILLS
# ============================================================

def normalize_candidate_skills(skills):
    """
    Convert extracted resume skills into a normalized set.
    """

    if skills is None:
        return set()

    if isinstance(skills, str):

        # Support comma-separated strings
        skills = skills.split(",")

    normalized = set()

    for skill in skills:

        skill = normalize_skill(skill)

        if skill:
            normalized.add(skill)

    return normalized


# ============================================================
# CALCULATE SKILL MATCH
# ============================================================

def calculate_skill_match(
    candidate_skills,
    required_skills,
):
    """
    Calculate matched and missing skills.
    """

    matched = (
        candidate_skills &
        required_skills
    )

    missing = (
        required_skills -
        candidate_skills
    )

    if required_skills:

        percentage = (
            len(matched)
            /
            len(required_skills)
        ) * 100

    else:

        percentage = 0.0

    return (
        matched,
        missing,
        percentage,
    )


# ============================================================
# RECOMMEND CAREERS
# ============================================================

def recommend_careers(
    predictions,
    candidate_skills,
    career_skill_map,
    top_k=5,
):
    """
    Combine model prediction probabilities
    with skill matching.

    Final score:

        70% model probability
        30% skill match
    """

    candidate_skills = normalize_candidate_skills(
        candidate_skills
    )

    results = []

    for prediction in predictions:

        career = prediction["career"]

        probability = float(
            prediction["probability"]
        )

        required_skills = career_skill_map.get(
            career,
            set(),
        )

        (
            matched_skills,
            missing_skills,
            skill_match_percentage,
        ) = calculate_skill_match(
            candidate_skills,
            required_skills,
        )

        model_score = (
            probability * 100
        )

        final_score = (
            0.70 * model_score
            +
            0.30 * skill_match_percentage
        )

        results.append({

            "career": career,

            "model_probability":
                round(
                    probability,
                    6,
                ),

            "model_score":
                round(
                    model_score,
                    2,
                ),

            "skill_match_percentage":
                round(
                    skill_match_percentage,
                    2,
                ),

            "matched_skills":
                sorted(
                    matched_skills
                ),

            "missing_skills":
                sorted(
                    missing_skills
                ),

            "final_score":
                round(
                    final_score,
                    2,
                ),
        })

    results.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    results = results[:top_k]

    for rank, result in enumerate(
        results,
        start=1,
    ):

        result["rank"] = rank

    return results


# ============================================================
# RECOMMEND USING ALL THREE MODELS
# ============================================================

def generate_recommendations(
    prediction_result,
    candidate_skills,
    top_k=5,
):
    """
    Generate recommendations for each
    verified Milestone 2 classifier.
    """

    profiles = load_career_profiles()

    career_skill_map = (
        build_career_skill_map(
            profiles
        )
    )

    model_results = {}

    for model_name, predictions in (
        prediction_result["models"].items()
    ):

        model_results[model_name] = (
            recommend_careers(
                predictions,
                candidate_skills,
                career_skill_map,
                top_k=top_k,
            )
        )

    return model_results
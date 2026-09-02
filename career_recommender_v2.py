# -*- coding: utf-8 -*-

"""
CAREERCAST MILESTONE 2
STEP 6 - V2 CAREER RECOMMENDATION RANKER

Purpose:
    Improve final career ranking using:
        1. Candidate skill-to-career skill match
        2. Sentence-BERT semantic similarity
        3. Existing XGBoost probability

IMPORTANT:
    - Existing XGBoost model is NOT modified.
    - Existing Sentence-BERT artifacts are NOT modified.
    - Existing dataset is NOT modified.
    - Existing recommendation files are NOT overwritten.
    - This script creates a new V2 output directory.
"""

import os
import re
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "results",
    "careercast_candidate_profiles.csv"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "results",
    "tree_features",
    "tfidf_vectorizer.joblib"
)

SVD_PATH = os.path.join(
    BASE_DIR,
    "results",
    "tree_features",
    "svd.joblib"
)

RIASEC_SCALER_PATH = os.path.join(
    BASE_DIR,
    "results",
    "tree_features",
    "riasec_scaler.joblib"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "results",
    "tree_features",
    "label_encoder.joblib"
)

XGB_MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "xgboost_model",
    "xgboost_model.joblib"
)

CAREER_EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "semantic_embeddings",
    "career_embeddings.npy"
)

CAREER_LABELS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "semantic_embeddings",
    "career_labels.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "final_recommendations_v2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# WEIGHTS
# ============================================================

# Skill relevance is intentionally strongest.
SKILL_WEIGHT = 0.50
SEMANTIC_WEIGHT = 0.30
XGB_WEIGHT = 0.20


# ============================================================
# HELPERS
# ============================================================

def log(message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


def check_file(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required {description} not found:\n{path}"
        )


def normalize_text(text):
    """
    Normalize text for matching.

    This does not change the original dataset.
    It is only used internally for comparison.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize common punctuation.
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # Remove brackets but preserve their content.
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)

    # Keep letters, numbers and spaces.
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_skill(skill):
    """
    Normalize a single skill.
    """
    skill = normalize_text(skill)

    # Common SQL typo/variation.
    if skill == "structure query language sql":
        return "sql"

    # Common naming variations.
    replacements = {
        "numpy": "numpy",
        "pandas": "pandas",
        "tensorflow": "tensorflow",
        "machine learning": "machine learning",
        "deep learning": "deep learning",
        "python": "python",
        "sql": "sql",
        "data analysis": "data analysis",
        "data analytics": "data analysis",
        "artificial intelligence": "artificial intelligence",
        "ai": "artificial intelligence",
    }

    return replacements.get(skill, skill)


def split_skills(text):
    """
    Split a comma-separated Skills field into normalized skills.
    """
    if pd.isna(text):
        return []

    raw_parts = str(text).split(",")

    result = []

    for part in raw_parts:
        part = normalize_skill(part)

        if part:
            result.append(part)

    return sorted(set(result))


def build_skill_match(candidate_skills, career_skills):
    """
    Calculate a transparent skill-overlap score.

    Exact matches are strongest.

    We also recognize a small number of carefully defined
    related skill groups so that obvious equivalents can
    contribute without making the recommender artificially
    broad.
    """

    candidate = set(candidate_skills)
    career = set(career_skills)

    if not candidate or not career:
        return 0.0, [], []


    # --------------------------------------------------------
    # Exact matches
    # --------------------------------------------------------

    exact_matches = sorted(candidate.intersection(career))


    # --------------------------------------------------------
    # Carefully controlled related groups
    # --------------------------------------------------------

    related_groups = [
        {
            "python",
            "python programming",
            "python software"
        },
        {
            "sql",
            "structure query language sql",
            "sql server"
        },
        {
            "machine learning",
            "machine learning software"
        },
        {
            "deep learning",
            "deep learning software"
        },
        {
            "tensorflow",
            "tensorflow software"
        },
        {
            "pandas",
            "pandas software"
        },
        {
            "numpy",
            "numpy software"
        },
        {
            "data analysis",
            "data analytics",
            "statistical analysis software",
            "statistical analysis"
        },
        {
            "database software",
            "database management software"
        },
    ]


    related_matches = set()

    for group in related_groups:

        candidate_has_group = any(
            item in candidate for item in group
        )

        career_has_group = any(
            item in career for item in group
        )

        if candidate_has_group and career_has_group:

            # Don't double count exact matches.
            for item in group:
                if item in candidate and item in career:
                    continue

            related_matches.add(
                next(
                    (
                        item
                        for item in group
                        if item in career
                    ),
                    next(iter(group))
                )
            )


    # --------------------------------------------------------
    # Weighted matching
    # --------------------------------------------------------

    exact_count = len(exact_matches)
    related_count = len(related_matches)

    # Exact match = 1.0
    # Related match = 0.50
    weighted_matches = exact_count + (0.50 * related_count)

    # Normalize by number of candidate skills.
    score = weighted_matches / max(len(candidate), 1)

    # Keep within [0,1].
    score = float(np.clip(score, 0.0, 1.0))

    all_matches = exact_matches + sorted(related_matches)

    # Suggested gaps are career skills not present in candidate.
    gaps = sorted(
        career.difference(candidate)
    )

    return score, all_matches, gaps


def minmax_normalize(values):
    """
    Min-max normalization.
    """
    values = np.asarray(values, dtype=float)

    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmax, vmin):
        return np.zeros_like(values)

    return (values - vmin) / (vmax - vmin)


def parse_education_to_numeric(text):
    """
    Convert degree text into a training-compatible approximate
    numeric representation.

    The original training data uses Education values around
    7.75-8.93.

    This is ONLY for XGBoost compatibility.
    """

    text = normalize_text(text)

    if "phd" in text or "doctor" in text:
        return 8.90

    if "master" in text or "m.e" in text or "m.tech" in text:
        return 8.70

    if "b.e" in text or "b.tech" in text:
        return 8.50

    if "b.sc" in text or "bca" in text:
        return 8.30

    if "diploma" in text:
        return 8.00

    if "12" in text or "higher secondary" in text:
        return 7.85

    # Safe fallback inside observed training range.
    return 8.30


def parse_experience_to_years(text):
    """
    Convert natural-language experience into years.
    """

    text = normalize_text(text)

    # Months
    month_match = re.search(
        r"(\d+(?:\.\d+)?)\s*month",
        text
    )

    if month_match:
        months = float(month_match.group(1))
        return months / 12.0

    # Years
    year_match = re.search(
        r"(\d+(?:\.\d+)?)\s*year",
        text
    )

    if year_match:
        return float(year_match.group(1))

    # Fresher
    if "fresher" in text or "no experience" in text:
        return 0.0

    return 0.0


def build_xgb_features(
    skills,
    education_numeric,
    experience_numeric,
    riasec_values,
    tfidf,
    svd,
    riasec_scaler
):

    combined_text = (
        f"{skills} "
        f"{education_numeric} "
        f"{experience_numeric}"
    )

    tfidf_vector = tfidf.transform(
        [combined_text]
    )

    svd_vector = svd.transform(
        tfidf_vector
    )

    riasec_array = np.array(
        [riasec_values],
        dtype=float
    )

    scaled_riasec = riasec_scaler.transform(
        riasec_array
    )

    features = np.hstack(
        [
            svd_vector,
            scaled_riasec
        ]
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()
    print("=" * 75)
    print("CAREERCAST MILESTONE 2")
    print("STEP 6 - V2 CAREER RECOMMENDATION RANKER")
    print("=" * 75)

    print()
    print("READ-ONLY MODEL MODE")
    print("Existing XGBoost model will NOT be modified.")
    print("Existing SBERT embeddings will NOT be modified.")
    print("Existing dataset will NOT be modified.")
    print("Existing recommendation files will NOT be overwritten.")

    # ========================================================
    # STAGE 1
    # ========================================================

    print()
    print("Stage 1/8  Checking artifacts...")

    required_files = [
        (DATASET_PATH, "dataset"),
        (TFIDF_PATH, "TF-IDF vectorizer"),
        (SVD_PATH, "SVD model"),
        (RIASEC_SCALER_PATH, "RIASEC scaler"),
        (LABEL_ENCODER_PATH, "label encoder"),
        (XGB_MODEL_PATH, "XGBoost model"),
        (CAREER_EMBEDDINGS_PATH, "career embeddings"),
        (CAREER_LABELS_PATH, "career labels"),
    ]

    for path, description in required_files:
        check_file(path, description)

    print("All required artifacts found.")

    # ========================================================
    # STAGE 2
    # ========================================================

    print()
    print("Stage 2/8  Loading artifacts...")

    dataset = pd.read_csv(DATASET_PATH)

    tfidf = joblib.load(TFIDF_PATH)
    svd = joblib.load(SVD_PATH)
    riasec_scaler = joblib.load(RIASEC_SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)

    career_embeddings = np.load(
        CAREER_EMBEDDINGS_PATH
    )

    career_labels_df = pd.read_csv(
        CAREER_LABELS_PATH
    )

    print(f"Dataset rows       : {len(dataset)}")
    print(f"Dataset careers    : {dataset['Career'].nunique()}")
    print(f"TF-IDF vocabulary  : {len(tfidf.vocabulary_)}")
    print(f"SVD components     : {svd.n_components}")
    print(f"Career embeddings  : {career_embeddings.shape}")

    # ========================================================
    # STAGE 3
    # ========================================================

    print()
    print("Stage 3/8  Loading Sentence-BERT...")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )

    print("Model: all-MiniLM-L6-v2")

    semantic_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Sentence-BERT loaded successfully.")

    # ========================================================
    # STAGE 4
    # ========================================================

    print()
    print("Stage 4/8  Candidate input")

    print()
    print("Enter candidate information.")
    print()

    skills_text = input("Skills: ").strip()
    education_text = input("Education: ").strip()
    experience_text = input("Experience: ").strip()
    job_description = input("Job Description: ").strip()

    print()
    print("Enter the six RIASEC scores.")
    print("Use the same scale used in the training dataset.")

    realistic = float(input("Realistic: "))
    investigative = float(input("Investigative: "))
    artistic = float(input("Artistic: "))
    social = float(input("Social: "))
    enterprising = float(input("Enterprising: "))
    conventional = float(input("Conventional: "))

    riasec_values = [
        realistic,
        investigative,
        artistic,
        social,
        enterprising,
        conventional,
    ]

    # --------------------------------------------------------
    # Training-compatible numeric representations
    # --------------------------------------------------------

    education_numeric = parse_education_to_numeric(
        education_text
    )

    experience_numeric = parse_experience_to_years(
        experience_text
    )

    print()
    print("=" * 75)
    print("TRAINING-COMPATIBLE CANDIDATE REPRESENTATION")
    print("=" * 75)

    print(
        f"Education text : {education_text}"
    )

    print(
        f"Education used : {education_numeric:.4f}"
    )

    print(
        f"Experience text: {experience_text}"
    )

    print(
        f"Experience used: {experience_numeric:.4f} years"
    )

    # ========================================================
    # STAGE 5
    # ========================================================

    print()
    print("Stage 5/8  Building XGBoost features...")

    xgb_features = build_xgb_features(
        skills_text,
        education_numeric,
        experience_numeric,
        riasec_values,
        tfidf,
        svd,
        riasec_scaler
    )

    print(
        f"XGBoost feature shape: {xgb_features.shape}"
    )

    if xgb_features.shape[1] != 206:
        raise ValueError(
            f"Expected 206 XGBoost features, "
            f"got {xgb_features.shape[1]}"
        )

    # ========================================================
    # STAGE 6
    # ========================================================

    print()
    print("Stage 6/8  Running model inference...")

    # XGBoost
    xgb_probabilities = xgb_model.predict_proba(
        xgb_features
    )[0]

    # --------------------------------------------------------
    # Candidate semantic embedding
    # --------------------------------------------------------

    candidate_profile_text = (
        f"Skills: {skills_text}. "
        f"Education: {education_text}. "
        f"Experience: {experience_text}. "
        f"Job Description: {job_description}. "
        f"RIASEC: {riasec_values}."
    )

    candidate_embedding = semantic_model.encode(
        [candidate_profile_text],
        normalize_embeddings=True
    )[0]

    career_embeddings_normalized = career_embeddings

    # Ensure normalization.
    norms = np.linalg.norm(
        career_embeddings_normalized,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    career_embeddings_normalized = (
        career_embeddings_normalized / norms
    )

    semantic_scores = np.dot(
        career_embeddings_normalized,
        candidate_embedding
    )

    # --------------------------------------------------------
    # Map XGBoost classes
    # --------------------------------------------------------

    xgb_classes = list(
        label_encoder.classes_
    )

    xgb_score_map = {
        career: float(prob)
        for career, prob
        in zip(
            xgb_classes,
            xgb_probabilities
        )
    }

    # ========================================================
    # STAGE 7
    # ========================================================

    print()
    print("Stage 7/8  Building skill-aware ranking...")

    candidate_skills = split_skills(
        skills_text
    )

    print()
    print("Normalized candidate skills:")
    print(candidate_skills)

    # --------------------------------------------------------
    # Build one skill list per career
    # --------------------------------------------------------

    career_skill_map = {}

    for _, row in dataset.iterrows():

        career = str(row["Career"])

        if career not in career_skill_map:

            career_skill_map[career] = []

        career_skill_map[career].extend(
            split_skills(row["Skills"])
        )

    # Remove duplicates.
    for career in career_skill_map:

        career_skill_map[career] = sorted(
            set(career_skill_map[career])
        )

    # --------------------------------------------------------
    # Create scores
    # --------------------------------------------------------

    records = []

    for idx, career in enumerate(
        career_labels_df.iloc[:, 0].astype(str)
    ):

        career_skills = career_skill_map.get(
            career,
            []
        )

        skill_score, matches, gaps = (
            build_skill_match(
                candidate_skills,
                career_skills
            )
        )

        semantic_raw = float(
            semantic_scores[idx]
        )

        xgb_raw = float(
            xgb_score_map.get(
                career,
                0.0
            )
        )

        records.append(
            {
                "Career": career,
                "Skill_Match_Raw": skill_score,
                "Semantic_Raw": semantic_raw,
                "XGBoost_Raw": xgb_raw,
                "Matched_Skills": ", ".join(matches),
                "Suggested_Skill_Gaps": ", ".join(
                    gaps[:15]
                ),
                "Matched_Skill_Count": len(matches),
            }
        )

    ranking_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Normalize semantic and XGB signals
    # --------------------------------------------------------

    ranking_df["Skill_Match_Normalized"] = (
        ranking_df["Skill_Match_Raw"]
    )

    ranking_df["Semantic_Normalized"] = (
        minmax_normalize(
            ranking_df["Semantic_Raw"]
        )
    )

    ranking_df["XGBoost_Normalized"] = (
        minmax_normalize(
            ranking_df["XGBoost_Raw"]
        )
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    ranking_df["Final_Score"] = (
        SKILL_WEIGHT
        * ranking_df["Skill_Match_Normalized"]
        +
        SEMANTIC_WEIGHT
        * ranking_df["Semantic_Normalized"]
        +
        XGB_WEIGHT
        * ranking_df["XGBoost_Normalized"]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranking_df = ranking_df.sort_values(
        by=[
            "Final_Score",
            "Skill_Match_Normalized",
            "Semantic_Normalized"
        ],
        ascending=False
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(ranking_df) + 1
        )
    )

    # ========================================================
    # STAGE 8
    # ========================================================

    print()
    print("Stage 8/8  Generating V2 recommendation report...")

    print()
    print("=" * 75)
    print("CAREERCAST - V2 AI CAREER RECOMMENDATION")
    print("=" * 75)

    print()
    print("SCORING")
    print("-" * 75)

    print(
        f"Skill Match Weight : {SKILL_WEIGHT:.2f}"
    )

    print(
        f"Semantic Weight    : {SEMANTIC_WEIGHT:.2f}"
    )

    print(
        f"XGBoost Weight     : {XGB_WEIGHT:.2f}"
    )

    print()
    print("CANDIDATE")
    print("-" * 75)

    print(f"Skills     : {skills_text}")
    print(f"Education  : {education_text}")
    print(
        f"Education numeric representation: "
        f"{education_numeric:.2f}"
    )

    print(f"Experience : {experience_text}")
    print(
        f"Experience numeric years: "
        f"{experience_numeric:.2f}"
    )

    print(f"Job Desc.  : {job_description}")

    print()
    print("RIASEC")

    labels = [
        "Realistic",
        "Investigative",
        "Artistic",
        "Social",
        "Enterprising",
        "Conventional",
    ]

    for label, value in zip(
        labels,
        riasec_values
    ):
        print(
            f"  {label:<15}: {value:.1f}"
        )

    print()
    print("=" * 75)
    print("TOP 10 CAREER RECOMMENDATIONS")
    print("=" * 75)

    for _, row in ranking_df.head(10).iterrows():

        print()
        print(
            f"{int(row['Rank'])}. "
            f"{row['Career']}"
        )

        print(
            f"   Skill Match Score    : "
            f"{row['Skill_Match_Normalized']:.4f}"
        )

        print(
            f"   Semantic Similarity  : "
            f"{row['Semantic_Raw']:.4f}"
        )

        print(
            f"   XGBoost Probability  : "
            f"{row['XGBoost_Raw']:.6f}"
        )

        print(
            f"   FINAL SCORE          : "
            f"{row['Final_Score']:.4f}"
        )

        if row["Matched_Skills"]:
            print(
                f"   Matched Skills       : "
                f"{row['Matched_Skills']}"
            )
        else:
            print(
                "   Matched Skills       : "
                "None identified"
            )

        if row["Suggested_Skill_Gaps"]:
            print(
                f"   Suggested Skill Gaps : "
                f"{row['Suggested_Skill_Gaps']}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"recommendations_v2_{timestamp}.csv"
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        f"recommendation_v2_summary_{timestamp}.json"
    )

    ranking_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "project": "CareerCast",
        "milestone": 2,
        "step": 6,
        "pipeline": "V2 skill-aware recommendation ranking",
        "timestamp": timestamp,
        "candidate": {
            "skills": skills_text,
            "education": education_text,
            "education_numeric": education_numeric,
            "experience": experience_text,
            "experience_years": experience_numeric,
            "job_description": job_description,
            "riasec": {
                "Realistic": realistic,
                "Investigative": investigative,
                "Artistic": artistic,
                "Social": social,
                "Enterprising": enterprising,
                "Conventional": conventional,
            },
        },
        "weights": {
            "skill_match": SKILL_WEIGHT,
            "semantic": SEMANTIC_WEIGHT,
            "xgboost": XGB_WEIGHT,
        },
        "model_protection": {
            "xgboost_retrained": False,
            "xgboost_modified": False,
            "sbert_modified": False,
            "dataset_modified": False,
        },
        "top_10": ranking_df.head(10)[
            [
                "Rank",
                "Career",
                "Skill_Match_Normalized",
                "Semantic_Raw",
                "XGBoost_Raw",
                "Final_Score",
                "Matched_Skills",
                "Suggested_Skill_Gaps",
            ]
        ].to_dict(
            orient="records"
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    elapsed = time.time() - start_time

    print()
    print("=" * 75)
    print("V2 RECOMMENDATION COMPLETE")
    print("=" * 75)

    print()
    print("Full ranking saved to:")
    print(csv_path)

    print()
    print("Summary saved to:")
    print(json_path)

    print()
    print(
        f"Total time: {elapsed:.2f}s"
    )

    print()
    print("IMPORTANT:")
    print("- Existing XGBoost model was NOT retrained.")
    print("- Existing model artifacts were NOT modified.")
    print("- Existing recommendation files were NOT overwritten.")
    print("- New results were saved under:")
    print("  results/final_recommendations_v2/")
    print("=" * 75)


if __name__ == "__main__":
    main()
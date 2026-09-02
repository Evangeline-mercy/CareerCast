"""
# -*- coding: utf-8 -*-
CareerCast - Milestone 2
Step 5: Final Career Recommendation / Inference Pipeline

Uses existing trained artifacts only.

Pipeline:

Candidate Input
    |
    +-- Skills
    +-- Education
    +-- Experience
    +-- Job Description
    +-- RIASEC scores
    |
    +------------------------------+
    |                              |
    v                              v
XGBoost Pipeline             Sentence-BERT
    |                              |
TF-IDF -> SVD -> RIASEC      384-D embedding
    |                              |
    v                              v
XGBoost probabilities       Career semantic similarity
    |                              |
    +-------------+----------------+
                  |
                  v
          Final Career Ranking
                  |
                  v
             Top 10 Careers
                  |
        +---------+---------+
        |                   |
        v                   v
   Skill Matching      Explanation
   / Skill Gaps

IMPORTANT:
- No model retraining.
- No new dataset.
- No fabricated RIASEC values.
- RIASEC scores must be supplied by the user.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "results" / "careercast_candidate_profiles.csv"

TREE_DIR = BASE_DIR / "results" / "tree_features"
XGB_DIR = BASE_DIR / "results" / "xgboost_model"
SEMANTIC_DIR = BASE_DIR / "results" / "semantic_embeddings"
HYBRID_DIR = BASE_DIR / "results" / "hybrid_recommender"


# Tree preprocessing artifacts
TFIDF_PATH = TREE_DIR / "tfidf_vectorizer.joblib"
SVD_PATH = TREE_DIR / "svd.joblib"
RIASEC_SCALER_PATH = TREE_DIR / "riasec_scaler.joblib"
TREE_LABEL_ENCODER_PATH = TREE_DIR / "label_encoder.joblib"

# XGBoost artifacts
XGB_MODEL_PATH = XGB_DIR / "xgboost_model.joblib"
XGB_LABEL_ENCODER_PATH = XGB_DIR / "label_encoder.joblib"

# Semantic artifacts
CAREER_EMBEDDINGS_PATH = (
    SEMANTIC_DIR / "career_embeddings.npy"
)

CAREER_LABELS_PATH = (
    SEMANTIC_DIR / "career_labels.csv"
)

PROFILE_METADATA_PATH = (
    SEMANTIC_DIR / "profile_metadata.csv"
)

# Hybrid artifacts
ALPHA_RESULTS_PATH = (
    HYBRID_DIR / "alpha_tuning_results.csv"
)

HYBRID_METADATA_PATH = (
    HYBRID_DIR / "hybrid_recommender_metadata.json"
)

# Sentence-BERT
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Expected project dimensions
EXPECTED_CAREERS = 878
EXPECTED_TREE_FEATURES = 206
EXPECTED_SEMANTIC_DIM = 384

TOP_K = 10

RIASEC_COLUMNS = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True,
    )


def fail(message: str) -> None:
    print()
    print("=" * 75)
    print("ERROR")
    print("=" * 75)
    print(message)
    print("=" * 75)
    sys.exit(1)


# ============================================================================
# ARTIFACT VALIDATION
# ============================================================================

def check_required_files() -> None:
    """Verify all required artifacts exist."""

    required_files = [
        DATA_PATH,
        TFIDF_PATH,
        SVD_PATH,
        RIASEC_SCALER_PATH,
        TREE_LABEL_ENCODER_PATH,
        XGB_MODEL_PATH,
        XGB_LABEL_ENCODER_PATH,
        CAREER_EMBEDDINGS_PATH,
        CAREER_LABELS_PATH,
        PROFILE_METADATA_PATH,
        ALPHA_RESULTS_PATH,
        HYBRID_METADATA_PATH,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        message = (
            "The following required files are missing:\n\n"
            + "\n".join(missing)
        )
        fail(message)


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_artifacts() -> Dict:
    """Load all previously trained/saved artifacts."""

    log("Loading CareerCast artifacts...")

    check_required_files()

    artifacts = {}

    # Dataset
    artifacts["dataset"] = pd.read_csv(DATA_PATH)

    # Tree preprocessing
    artifacts["tfidf"] = joblib.load(TFIDF_PATH)
    artifacts["svd"] = joblib.load(SVD_PATH)
    artifacts["riasec_scaler"] = joblib.load(
        RIASEC_SCALER_PATH
    )

    # Label encoders
    artifacts["tree_label_encoder"] = joblib.load(
        TREE_LABEL_ENCODER_PATH
    )

    artifacts["xgb_label_encoder"] = joblib.load(
        XGB_LABEL_ENCODER_PATH
    )

    # XGBoost
    artifacts["xgb_model"] = joblib.load(
        XGB_MODEL_PATH
    )

    # Semantic career embeddings
    artifacts["career_embeddings"] = np.load(
        CAREER_EMBEDDINGS_PATH
    ).astype(np.float32)

    artifacts["career_labels"] = pd.read_csv(
        CAREER_LABELS_PATH
    )

    # Profile metadata
    artifacts["profile_metadata"] = pd.read_csv(
        PROFILE_METADATA_PATH
    )

    # Hybrid information
    artifacts["alpha_results"] = pd.read_csv(
        ALPHA_RESULTS_PATH
    )

    with open(
        HYBRID_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        artifacts["hybrid_metadata"] = json.load(f)

    log("All artifacts loaded successfully.")

    return artifacts


# ============================================================================
# VALIDATION
# ============================================================================

def validate_artifacts(artifacts: Dict) -> None:
    """Validate shapes and career ordering."""

    log("Validating artifact compatibility...")

    df = artifacts["dataset"]

    if len(df) != 17560:
        print(
            f"WARNING: Dataset has {len(df)} rows; "
            f"expected 17560."
        )

    # ------------------------------------------------------------------
    # Dataset columns
    # ------------------------------------------------------------------

    required_columns = [
        "Profile_ID",
        "Career",
        "Skills",
        "Education",
        "Experience",
        "Job_Description",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        fail(
            "Dataset is missing required columns: "
            + str(missing_columns)
        )

    # RIASEC columns may exist in dataset
    # but are not required because the user enters them manually.

    # ------------------------------------------------------------------
    # Career embeddings
    # ------------------------------------------------------------------

    career_embeddings = artifacts["career_embeddings"]
    career_labels = artifacts["career_labels"]

    if career_embeddings.ndim != 2:
        fail(
            "career_embeddings.npy must be a 2-D matrix."
        )

    if career_embeddings.shape != (
        EXPECTED_CAREERS,
        EXPECTED_SEMANTIC_DIM,
    ):
        fail(
            "Unexpected career embedding shape.\n"
            f"Found: {career_embeddings.shape}\n"
            f"Expected: "
            f"({EXPECTED_CAREERS}, "
            f"{EXPECTED_SEMANTIC_DIM})"
        )

    if len(career_labels) != EXPECTED_CAREERS:
        fail(
            f"Expected {EXPECTED_CAREERS} career labels, "
            f"found {len(career_labels)}."
        )

    if "Career" not in career_labels.columns:
        fail(
            "career_labels.csv does not contain "
            "a 'Career' column."
        )

    # ------------------------------------------------------------------
    # XGBoost classes
    # ------------------------------------------------------------------

    xgb_model = artifacts["xgb_model"]

    try:
        xgb_probabilities = xgb_model.predict_proba(
            np.zeros(
                (1, EXPECTED_TREE_FEATURES),
                dtype=np.float32,
            )
        )

        if xgb_probabilities.shape[1] != EXPECTED_CAREERS:
            fail(
                "XGBoost does not output the expected "
                f"{EXPECTED_CAREERS} classes.\n"
                f"Found: {xgb_probabilities.shape[1]}"
            )

    except Exception as exc:
        fail(
            "Could not validate XGBoost model output.\n"
            f"Details: {exc}"
        )

    # ------------------------------------------------------------------
    # Label encoder validation
    # ------------------------------------------------------------------

    tree_classes = list(
        artifacts["tree_label_encoder"].classes_
    )

    xgb_classes = list(
        artifacts["xgb_label_encoder"].classes_
    )

    semantic_classes = list(
        career_labels["Career"].astype(str)
    )

    if len(tree_classes) != EXPECTED_CAREERS:
        fail(
            "Tree label encoder does not contain "
            f"{EXPECTED_CAREERS} classes."
        )

    if len(xgb_classes) != EXPECTED_CAREERS:
        fail(
            "XGBoost label encoder does not contain "
            f"{EXPECTED_CAREERS} classes."
        )

    if set(tree_classes) != set(xgb_classes):
        fail(
            "Tree and XGBoost label encoders contain "
            "different career sets."
        )

    if set(xgb_classes) != set(semantic_classes):
        fail(
            "XGBoost career labels and semantic career "
            "labels do not contain the same career set."
        )

    # ------------------------------------------------------------------
    # Semantic embedding normalization
    # ------------------------------------------------------------------

    norms = np.linalg.norm(
        career_embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3,
    ):
        print(
            "WARNING: Career embeddings are not perfectly "
            "L2-normalized. They will be normalized during inference."
        )

    log("Artifact validation successful.")


# ============================================================================
# HYBRID ALPHA
# ============================================================================

def determine_alpha(
    artifacts: Dict,
) -> float:
    """
    Determine validated alpha.

    Preference:
    1. Existing metadata if it explicitly contains alpha.
    2. Alpha sweep based on highest Top-1 accuracy.
    3. MRR as tie breaker.

    This avoids inventing an alpha.
    """

    metadata = artifacts["hybrid_metadata"]

    possible_keys = [
        "best_alpha",
        "selected_alpha",
        "optimal_alpha",
        "alpha",
    ]

    for key in possible_keys:
        if key in metadata:
            try:
                alpha = float(metadata[key])

                if 0.0 <= alpha <= 1.0:
                    log(
                        f"Using alpha from metadata: "
                        f"{alpha:.2f}"
                    )
                    return alpha

            except (TypeError, ValueError):
                pass

    results = artifacts["alpha_results"].copy()

    required = {
        "alpha",
        "top1_accuracy",
        "mrr",
    }

    if not required.issubset(results.columns):
        log(
            "Could not determine validated alpha "
            "from existing sweep. Using alpha=1.0."
        )
        return 1.0

    results = results.sort_values(
        by=[
            "top1_accuracy",
            "mrr",
        ],
        ascending=[
            False,
            False,
        ],
    )

    alpha = float(results.iloc[0]["alpha"])

    log(
        f"Best validated alpha from sweep: "
        f"{alpha:.2f}"
    )

    return alpha


# ============================================================================
# INPUT HELPERS
# ============================================================================

def ask_text(
    label: str,
    allow_empty: bool = False,
) -> str:
    """Get text input from user."""

    while True:
        value = input(f"{label}: ").strip()

        if value or allow_empty:
            return value

        print(
            "Input cannot be empty. Please enter a value."
        )


def ask_float(
    label: str,
) -> float:
    """Get numeric RIASEC input."""

    while True:
        raw = input(f"{label}: ").strip()

        try:
            value = float(raw)

            if not np.isfinite(value):
                raise ValueError

            return value

        except ValueError:
            print(
                "Please enter a valid numeric value."
            )


def collect_candidate() -> Dict:
    """Collect a new candidate profile."""

    print()
    print("=" * 75)
    print("CAREERCAST � NEW CANDIDATE PROFILE")
    print("=" * 75)

    print()
    print("Enter candidate information.")
    print()

    skills = ask_text("Skills")
    education = ask_text("Education")
    experience = ask_text("Experience")
    job_description = ask_text(
        "Job Description"
    )

    print()
    print(
        "Enter the six RIASEC scores used by the "
        "existing model."
    )
    print(
        "Use the same scoring scale used when "
        "creating the training dataset."
    )
    print()

    riasec = {}

    for column in RIASEC_COLUMNS:
        riasec[column] = ask_float(column)

    return {
        "Skills": skills,
        "Education": education,
        "Experience": experience,
        "Job_Description": job_description,
        "RIASEC": riasec,
    }


# ============================================================================
# XGBOOST FEATURE PREPARATION
# ============================================================================

def build_tree_features(
    candidate: Dict,
    artifacts: Dict,
) -> np.ndarray:
    """
    Reproduce the existing tree feature pipeline:

    Skills + Education + Experience
        -> TF-IDF
        -> SVD
        -> RIASEC scaling
        -> concatenate
        -> 206 features
    """

    text_blob = (
        str(candidate["Skills"])
        + " . "
        + str(candidate["Education"])
        + " . "
        + str(candidate["Experience"])
    )

    tfidf = artifacts["tfidf"]
    svd = artifacts["svd"]
    scaler = artifacts["riasec_scaler"]

    # ------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------

    tfidf_features = tfidf.transform(
        [text_blob]
    )

    # ------------------------------------------------------------
    # SVD
    # ------------------------------------------------------------

    svd_features = svd.transform(
        tfidf_features
    ).astype(np.float32)

    if svd_features.shape[1] != 200:
        fail(
            "The loaded SVD does not produce "
            f"200 features. Found {svd_features.shape[1]}."
        )

    # ------------------------------------------------------------
    # RIASEC
    # ------------------------------------------------------------

    riasec_values = np.array(
        [
            candidate["RIASEC"][column]
            for column in RIASEC_COLUMNS
        ],
        dtype=np.float32,
    ).reshape(1, -1)

    scaled_riasec = scaler.transform(
        riasec_values
    ).astype(np.float32)

    if scaled_riasec.shape[1] != 6:
        fail(
            "RIASEC scaler did not produce 6 features."
        )

    # ------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------

    features = np.hstack(
        [
            svd_features,
            scaled_riasec,
        ]
    ).astype(np.float32)

    if features.shape != (
        1,
        EXPECTED_TREE_FEATURES,
    ):
        fail(
            "Final XGBoost feature vector has the "
            f"wrong shape: {features.shape}. "
            f"Expected (1, {EXPECTED_TREE_FEATURES})."
        )

    return features


# ============================================================================
# XGBOOST PREDICTION
# ============================================================================

def get_xgb_predictions(
    candidate: Dict,
    artifacts: Dict,
) -> pd.DataFrame:
    """Generate XGBoost probabilities mapped to career names."""

    features = build_tree_features(
        candidate,
        artifacts,
    )

    model = artifacts["xgb_model"]

    probabilities = model.predict_proba(
        features
    )[0].astype(np.float64)

    xgb_encoder = artifacts[
        "xgb_label_encoder"
    ]

    classes = list(
        xgb_encoder.classes_
    )

    if len(probabilities) != len(classes):
        fail(
            "XGBoost probability vector length "
            "does not match label encoder."
        )

    result = pd.DataFrame(
        {
            "Career": classes,
            "XGB_Probability": probabilities,
        }
    )

    result = result.sort_values(
        "XGB_Probability",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ============================================================================
# SEMANTIC PREDICTION
# ============================================================================

def build_semantic_candidate_text(
    candidate: Dict,
) -> str:
    """
    Build semantic input using:

    Skills
    Education
    Experience
    Job Description

    Career label is never included.
    """

    return (
        f"Skills: {candidate['Skills']} "
        f"Education: {candidate['Education']} "
        f"Experience: {candidate['Experience']} "
        f"Job Description: {candidate['Job_Description']}"
    ).strip()


def get_semantic_predictions(
    candidate: Dict,
    artifacts: Dict,
    model: SentenceTransformer,
) -> pd.DataFrame:
    """Calculate semantic similarity against all 878 careers."""

    text = build_semantic_candidate_text(
        candidate
    )

    candidate_embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    if candidate_embedding.shape != (
        1,
        EXPECTED_SEMANTIC_DIM,
    ):
        fail(
            "Unexpected Sentence-BERT embedding shape: "
            f"{candidate_embedding.shape}"
        )

    career_embeddings = (
        artifacts["career_embeddings"]
        .astype(np.float32)
    )

    # Normalize career embeddings defensively.
    norms = np.linalg.norm(
        career_embeddings,
        axis=1,
        keepdims=True,
    )

    career_embeddings = (
        career_embeddings
        / np.maximum(norms, 1e-12)
    )

    similarities = (
        career_embeddings
        @ candidate_embedding.T
    ).flatten()

    labels = (
        artifacts["career_labels"]["Career"]
        .astype(str)
        .tolist()
    )

    result = pd.DataFrame(
        {
            "Career": labels,
            "Semantic_Similarity": similarities,
        }
    )

    result = result.sort_values(
        "Semantic_Similarity",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ============================================================================
# SCORE NORMALIZATION
# ============================================================================

def minmax_normalize(
    values: np.ndarray,
) -> np.ndarray:
    """Normalize an array to [0, 1]."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-12:
        return np.zeros_like(values)

    return (
        (values - minimum)
        / (maximum - minimum)
    )


# ============================================================================
# MERGE / FINAL RANKING
# ============================================================================

def build_final_ranking(
    xgb_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """
    Align XGBoost and semantic results by Career.

    Then calculate:

    XGB normalized score
    Semantic normalized score
    Hybrid score
    """

    xgb_careers = set(
        xgb_df["Career"]
    )

    semantic_careers = set(
        semantic_df["Career"]
    )

    if xgb_careers != semantic_careers:
        fail(
            "XGBoost and semantic career sets "
            "do not match."
        )

    # Use explicit Career join to avoid row-order errors.
    merged = pd.merge(
        xgb_df,
        semantic_df,
        on="Career",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_CAREERS:
        fail(
            "Career alignment failed.\n"
            f"Expected {EXPECTED_CAREERS} careers, "
            f"got {len(merged)}."
        )

    # Normalize exactly as the hybrid approach requires.
    merged["XGB_Normalized"] = minmax_normalize(
        merged["XGB_Probability"].values
    )

    merged["Semantic_Normalized"] = minmax_normalize(
        merged["Semantic_Similarity"].values
    )

    merged["Hybrid_Score"] = (
        alpha * merged["XGB_Normalized"]
        + (1.0 - alpha)
        * merged["Semantic_Normalized"]
    )

    merged = merged.sort_values(
        [
            "Hybrid_Score",
            "XGB_Probability",
            "Semantic_Similarity",
        ],
        ascending=False,
    ).reset_index(drop=True)

    merged["Rank"] = (
        np.arange(len(merged)) + 1
    )

    return merged


# ============================================================================
# SKILL PROCESSING
# ============================================================================

def normalize_skill(skill: str) -> str:
    """Normalize a skill string for comparison."""

    skill = str(skill).strip().lower()

    skill = re.sub(
        r"\s+",
        " ",
        skill,
    )

    return skill


def split_skills(value: str) -> List[str]:
    """
    Split skill strings using common delimiters.

    Handles:
    comma
    semicolon
    pipe
    newline
    slash
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    parts = re.split(
        r"[,;|\n/]+",
        text,
    )

    skills = []

    for part in parts:
        normalized = normalize_skill(part)

        if normalized:
            skills.append(normalized)

    return skills


def candidate_skill_set(
    skills_text: str,
) -> set:
    return set(
        split_skills(skills_text)
    )


# ============================================================================
# DATASET-BASED SKILL GAP ANALYSIS
# ============================================================================

def get_career_skill_profile(
    career: str,
    dataset: pd.DataFrame,
) -> List[str]:
    """
    Extract representative skills for a career
    from the existing training dataset.

    No external skill database is used.
    """

    career_rows = dataset[
        dataset["Career"].astype(str) == str(career)
    ]

    if career_rows.empty:
        return []

    counts: Dict[str, int] = {}

    for value in career_rows["Skills"].tolist():
        for skill in split_skills(value):
            counts[skill] = (
                counts.get(skill, 0) + 1
            )

    # Sort by frequency, then alphabetically.
    ordered = sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    # Keep representative skills.
    return [
        skill
        for skill, _count in ordered[:15]
    ]


def calculate_skill_gap(
    candidate_skills: str,
    career: str,
    dataset: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Compare candidate skills against
    dataset-derived career skills.
    """

    candidate = candidate_skill_set(
        candidate_skills
    )

    career_skills = set(
        get_career_skill_profile(
            career,
            dataset,
        )
    )

    matched = sorted(
        candidate.intersection(career_skills)
    )

    missing = sorted(
        career_skills.difference(candidate)
    )

    return matched, missing


# ============================================================================
# RECOMMENDATION EXPLANATION
# ============================================================================

def build_reason(
    candidate: Dict,
    career: str,
    matched_skills: List[str],
    dataset: pd.DataFrame,
) -> str:
    """
    Create a conservative explanation based
    on candidate information and dataset skill match.
    """

    reasons = []

    if matched_skills:
        display_skills = ", ".join(
            matched_skills[:4]
        )

        reasons.append(
            f"matches dataset-associated skills "
            f"such as {display_skills}"
        )

    education = str(
        candidate["Education"]
    ).strip()

    if education:
        reasons.append(
            f"candidate education: {education}"
        )

    experience = str(
        candidate["Experience"]
    ).strip()

    if experience:
        reasons.append(
            "the supplied experience was included "
            "in the career prediction"
        )

    if not reasons:
        return (
            f"{career} was ranked highly by the "
            "trained CareerCast model."
        )

    return "; ".join(reasons) + "."


# ============================================================================
# DISPLAY
# ============================================================================

def print_candidate_summary(
    candidate: Dict,
) -> None:

    print()
    print("=" * 75)
    print("CANDIDATE SUMMARY")
    print("=" * 75)

    print(
        f"Skills      : {candidate['Skills']}"
    )

    print(
        f"Education   : {candidate['Education']}"
    )

    print(
        f"Experience  : {candidate['Experience']}"
    )

    print(
        f"Job Desc.   : {candidate['Job_Description']}"
    )

    print()
    print("RIASEC:")
    for key, value in candidate[
        "RIASEC"
    ].items():
        print(
            f"  {key:<15}: {value}"
        )


def print_results(
    ranking: pd.DataFrame,
    candidate: Dict,
    artifacts: Dict,
    alpha: float,
) -> None:

    top = ranking.head(TOP_K).copy()

    print()
    print("=" * 75)
    print("CAREERCAST � AI CAREER RECOMMENDATION")
    print("=" * 75)

    print()
    print(
        f"Validated XGBoost weight (alpha): "
        f"{alpha:.2f}"
    )

    if alpha >= 0.999:
        print(
            "Note: Current validation selects an "
            "XGBoost-dominant solution. "
            "Semantic ranking is retained as a "
            "supplementary signal."
        )

    print()
    print(
        "TOP 10 CAREER RECOMMENDATIONS"
    )
    print("=" * 75)

    for _, row in top.iterrows():

        career = str(
            row["Career"]
        )

        matched, missing = calculate_skill_gap(
            candidate["Skills"],
            career,
            artifacts["dataset"],
        )

        reason = build_reason(
            candidate,
            career,
            matched,
            artifacts["dataset"],
        )

        print()
        print(
            f"{int(row['Rank'])}. {career}"
        )

        print(
            f"   XGBoost Probability : "
            f"{row['XGB_Probability']:.4f}"
        )

        print(
            f"   Semantic Similarity : "
            f"{row['Semantic_Similarity']:.4f}"
        )

        print(
            f"   Hybrid Score        : "
            f"{row['Hybrid_Score']:.4f}"
        )

        print(
            f"   Why                 : "
            f"{reason}"
        )

        if matched:
            print(
                "   Matched Skills      : "
                + ", ".join(matched[:10])
            )
        else:
            print(
                "   Matched Skills      : None identified"
            )

        if missing:
            print(
                "   Suggested Skill Gaps: "
                + ", ".join(missing[:10])
            )
        else:
            print(
                "   Suggested Skill Gaps: None identified"
            )

    print()
    print("=" * 75)
    print("Recommendation complete.")
    print("=" * 75)


# ============================================================================
# OPTIONAL SAVE
# ============================================================================

def save_recommendations(
    ranking: pd.DataFrame,
) -> Path:
    """
    Save the complete ranking for the current candidate.
    """

    output_dir = (
        BASE_DIR
        / "results"
        / "final_recommendations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "latest_recommendations.csv"
    )

    ranking.to_csv(
        output_file,
        index=False,
    )

    return output_file


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    start = time.time()

    print()
    print("=" * 75)
    print("CAREERCAST MILESTONE 2")
    print("STEP 5 - FINAL CAREER RECOMMENDATION PIPELINE")
    print("=" * 75)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    log("Stage 1/7  Loading artifacts...")

    artifacts = load_artifacts()

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    log("Stage 2/7  Validating artifacts...")

    validate_artifacts(
        artifacts
    )

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------

    log("Stage 3/7  Loading Sentence-BERT...")

    try:
        semantic_model = SentenceTransformer(
            SBERT_MODEL_NAME
        )

    except Exception as exc:
        fail(
            "Could not load Sentence-BERT model.\n"
            f"Model: {SBERT_MODEL_NAME}\n"
            f"Details: {exc}"
        )

    log(
        "Sentence-BERT loaded successfully."
    )

    # ------------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------------

    log("Stage 4/7  Determining validated hybrid alpha...")

    alpha = determine_alpha(
        artifacts
    )

    # ------------------------------------------------------------------
    # Stage 5
    # ------------------------------------------------------------------

    log("Stage 5/7  Collecting candidate input...")

    candidate = collect_candidate()

    print_candidate_summary(
        candidate
    )

    # ------------------------------------------------------------------
    # Stage 6
    # ------------------------------------------------------------------

    log(
        "Stage 6/7  Running XGBoost + semantic inference..."
    )

    inference_start = time.time()

    xgb_results = get_xgb_predictions(
        candidate,
        artifacts,
    )

    semantic_results = get_semantic_predictions(
        candidate,
        artifacts,
        semantic_model,
    )

    ranking = build_final_ranking(
        xgb_results,
        semantic_results,
        alpha,
    )

    log(
        "Inference completed in "
        f"{time.time() - inference_start:.3f}s"
    )

    # ------------------------------------------------------------------
    # Stage 7
    # ------------------------------------------------------------------

    log("Stage 7/7  Generating recommendation report...")

    print_results(
        ranking,
        candidate,
        artifacts,
        alpha,
    )

    try:
        output_file = save_recommendations(
            ranking
        )

        print()
        print(
            f"Full ranking saved to:\n"
            f"{output_file}"
        )

    except Exception as exc:
        print(
            f"\nWARNING: Could not save ranking: {exc}"
        )

    total_time = time.time() - start

    print()
    print("=" * 75)
    print(
        f"Step 5 complete! "
        f"Total time: {total_time:.2f}s"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
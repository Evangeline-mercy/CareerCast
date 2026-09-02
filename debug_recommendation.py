# -*- coding: utf-8 -*-

"""
CareerCast Milestone 2
READ-ONLY DEBUG SCRIPT

Purpose:
    Diagnose why the final recommendation pipeline may produce
    unexpected career rankings.

IMPORTANT:
    - Does NOT retrain XGBoost.
    - Does NOT modify existing models.
    - Does NOT overwrite existing result files.
    - Only loads existing artifacts and performs inference.

Checks:
    1. XGBoost-only ranking
    2. Sentence-BERT-only ranking
    3. Hybrid ranking
    4. Career-label alignment
    5. Feature compatibility
    6. Hybrid alpha
"""

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "results/careercast_candidate_profiles.csv"

TREE_DIR = "results/tree_features"
XGB_DIR = "results/xgboost_model"
SEM_DIR = "results/semantic_embeddings"
HYBRID_DIR = "results/hybrid_recommender"


TFIDF_PATH = os.path.join(
    TREE_DIR, "tfidf_vectorizer.joblib"
)

SVD_PATH = os.path.join(
    TREE_DIR, "svd.joblib"
)

RIASEC_SCALER_PATH = os.path.join(
    TREE_DIR, "riasec_scaler.joblib"
)

TREE_LABEL_ENCODER_PATH = os.path.join(
    TREE_DIR, "label_encoder.joblib"
)

XGB_MODEL_PATH = os.path.join(
    XGB_DIR, "xgboost_model.joblib"
)

XGB_LABEL_ENCODER_PATH = os.path.join(
    XGB_DIR, "label_encoder.joblib"
)

CAREER_EMBEDDINGS_PATH = os.path.join(
    SEM_DIR, "career_embeddings.npy"
)

CAREER_LABELS_PATH = os.path.join(
    SEM_DIR, "career_labels.csv"
)

SEM_METADATA_PATH = os.path.join(
    SEM_DIR, "career_embedding_metadata.json"
)

HYBRID_METADATA_PATH = os.path.join(
    HYBRID_DIR, "hybrid_recommender_metadata.json"
)


MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 10

# This is the validated alpha reported by your Step 4 run.
ALPHA = 0.60


# ============================================================
# LOGGING
# ============================================================

def section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def log(message):
    print("[DEBUG] " + message)


# ============================================================
# SAFE FILE CHECK
# ============================================================

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():

    section("LOADING EXISTING ARTIFACTS")

    paths = [
        TFIDF_PATH,
        SVD_PATH,
        RIASEC_SCALER_PATH,
        TREE_LABEL_ENCODER_PATH,
        XGB_MODEL_PATH,
        XGB_LABEL_ENCODER_PATH,
        CAREER_EMBEDDINGS_PATH,
        CAREER_LABELS_PATH,
        SEM_METADATA_PATH,
    ]

    for path in paths:
        check_file(path)

    log("All required artifacts found.")

    tfidf = joblib.load(TFIDF_PATH)
    svd = joblib.load(SVD_PATH)
    riasec_scaler = joblib.load(RIASEC_SCALER_PATH)
    tree_label_encoder = joblib.load(TREE_LABEL_ENCODER_PATH)

    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_label_encoder = joblib.load(XGB_LABEL_ENCODER_PATH)

    career_embeddings = np.load(
        CAREER_EMBEDDINGS_PATH
    ).astype(np.float32)

    career_labels = pd.read_csv(
        CAREER_LABELS_PATH
    )

    with open(
        SEM_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        semantic_metadata = json.load(f)

    return (
        tfidf,
        svd,
        riasec_scaler,
        tree_label_encoder,
        xgb_model,
        xgb_label_encoder,
        career_embeddings,
        career_labels,
        semantic_metadata,
    )


# ============================================================
# CANDIDATE
# ============================================================

def get_candidate():

    section("CANDIDATE PROFILE")

    candidate = {
        "skills": (
            "Python, SQL, Machine Learning, Pandas, NumPy, "
            "Data Analysis, TensorFlow"
        ),

        "education": (
            "B.E. Electronics and Communication Engineering"
        ),

        "experience": (
            "6 months internship experience in Python, "
            "data analysis and machine learning projects"
        ),

        "job_description": (
            "Develop machine learning models, analyze datasets, "
            "build predictive systems, perform data preprocessing "
            "and visualization, and deploy data-driven applications."
        ),

        "Realistic": 6.0,
        "Investigative": 9.0,
        "Artistic": 4.0,
        "Social": 5.0,
        "Enterprising": 5.0,
        "Conventional": 8.0,
    }

    print("Skills      :", candidate["skills"])
    print("Education   :", candidate["education"])
    print("Experience  :", candidate["experience"])
    print("Job Desc.   :", candidate["job_description"])

    print()
    print("RIASEC:")
    print("  Realistic     :", candidate["Realistic"])
    print("  Investigative :", candidate["Investigative"])
    print("  Artistic      :", candidate["Artistic"])
    print("  Social        :", candidate["Social"])
    print("  Enterprising  :", candidate["Enterprising"])
    print("  Conventional  :", candidate["Conventional"])

    return candidate


# ============================================================
# TEXT FOR TREE MODEL
# ============================================================

def build_tree_text(candidate):

    return (
        str(candidate["skills"])
        + " . "
        + str(candidate["education"])
        + " . "
        + str(candidate["experience"])
    )


# ============================================================
# TEXT FOR SBERT
# ============================================================

def build_semantic_text(candidate):

    return (
        "Skills: "
        + str(candidate["skills"])
        + " Education: "
        + str(candidate["education"])
        + " Experience: "
        + str(candidate["experience"])
        + " Job Description: "
        + str(candidate["job_description"])
    ).strip()


# ============================================================
# XGBOOST FEATURE CREATION
# ============================================================

def build_xgb_features(
    candidate,
    tfidf,
    svd,
    riasec_scaler
):

    log("Creating XGBoost input features...")

    text = build_tree_text(candidate)

    tfidf_features = tfidf.transform([text])

    log(
        f"TF-IDF shape: {tfidf_features.shape}"
    )

    svd_features = svd.transform(
        tfidf_features
    )

    log(
        f"SVD shape: {svd_features.shape}"
    )

    riasec_values = np.array(
        [[
            candidate["Realistic"],
            candidate["Investigative"],
            candidate["Artistic"],
            candidate["Social"],
            candidate["Enterprising"],
            candidate["Conventional"],
        ]],
        dtype=np.float32
    )

    riasec_features = riasec_scaler.transform(
        riasec_values
    )

    log(
        f"RIASEC shape: {riasec_features.shape}"
    )

    X = np.hstack(
        [
            svd_features,
            riasec_features,
        ]
    )

    log(
        f"Final XGBoost feature shape: {X.shape}"
    )

    return X


# ============================================================
# XGBOOST RANKING
# ============================================================

def run_xgb(
    xgb_model,
    xgb_label_encoder,
    career_labels,
    X
):

    section("1. XGBOOST-ONLY RANKING")

    probabilities = xgb_model.predict_proba(X)

    print(
        "Probability shape:",
        probabilities.shape
    )

    classes = xgb_model.classes_

    print(
        "Number of XGBoost classes:",
        len(classes)
    )

    # Map XGBoost class index -> career name.
    try:
        model_careers = xgb_label_encoder.inverse_transform(
            classes.astype(int)
        )
    except Exception:
        model_careers = xgb_label_encoder.inverse_transform(
            np.asarray(classes, dtype=int)
        )

    ranking_indices = np.argsort(
        probabilities[0]
    )[::-1][:TOP_K]

    rows = []

    for rank, idx in enumerate(
        ranking_indices,
        start=1
    ):

        career = str(
            model_careers[idx]
        )

        probability = float(
            probabilities[0, idx]
        )

        rows.append(
            {
                "Rank": rank,
                "Career": career,
                "XGB_Probability": probability,
            }
        )

    result = pd.DataFrame(rows)

    print()
    print(
        f"{'Rank':<6}"
        f"{'Career':<55}"
        f"{'Probability':>12}"
    )

    print("-" * 75)

    for _, row in result.iterrows():

        print(
            f"{int(row['Rank']):<6}"
            f"{row['Career']:<55}"
            f"{row['XGB_Probability']:>12.6f}"
        )

    return probabilities, model_careers, result


# ============================================================
# SEMANTIC RANKING
# ============================================================

def run_semantic(
    candidate,
    career_embeddings,
    career_labels
):

    section("2. SENTENCE-BERT-ONLY RANKING")

    log(
        f"Loading Sentence-BERT: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    text = build_semantic_text(
        candidate
    )

    log("Encoding candidate profile...")

    candidate_embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    print(
        "Candidate embedding shape:",
        candidate_embedding.shape
    )

    similarities = (
        career_embeddings
        @ candidate_embedding.T
    ).flatten()

    indices = np.argsort(
        similarities
    )[::-1][:TOP_K]

    rows = []

    print()
    print(
        f"{'Rank':<6}"
        f"{'Career':<55}"
        f"{'Similarity':>12}"
    )

    print("-" * 75)

    for rank, idx in enumerate(
        indices,
        start=1
    ):

        career = str(
            career_labels.iloc[idx]["Career"]
        )

        similarity = float(
            similarities[idx]
        )

        rows.append(
            {
                "Rank": rank,
                "Career": career,
                "Semantic_Similarity": similarity,
            }
        )

        print(
            f"{rank:<6}"
            f"{career:<55}"
            f"{similarity:>12.6f}"
        )

    return similarities, result_from_rows(rows), model


def result_from_rows(rows):
    return pd.DataFrame(rows)


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    min_value = values.min()
    max_value = values.max()

    if np.isclose(
        max_value,
        min_value
    ):
        return np.zeros_like(
            values
        )

    return (
        (values - min_value)
        / (max_value - min_value)
    )


# ============================================================
# HYBRID RANKING
# ============================================================

def run_hybrid(
    probabilities,
    model_careers,
    semantic_similarities,
    career_labels,
    alpha
):

    section("3. HYBRID RANKING")

    log(
        f"Using alpha = {alpha:.2f}"
    )

    # --------------------------------------------------------
    # Create common career ordering
    # --------------------------------------------------------

    semantic_careers = (
        career_labels["Career"]
        .astype(str)
        .tolist()
    )

    xgb_map = {
        str(career): float(probabilities[0, i])
        for i, career in enumerate(model_careers)
    }

    semantic_map = {
        str(career): float(semantic_similarities[i])
        for i, career in enumerate(semantic_careers)
    }

    common_careers = sorted(
        set(xgb_map.keys())
        & set(semantic_map.keys())
    )

    print(
        "XGBoost careers   :",
        len(xgb_map)
    )

    print(
        "Semantic careers  :",
        len(semantic_map)
    )

    print(
        "Common careers    :",
        len(common_careers)
    )

    if len(common_careers) != 878:

        print()
        print(
            "WARNING: Career label alignment is NOT 878/878."
        )

    xgb_values = np.array(
        [
            xgb_map[c]
            for c in common_careers
        ]
    )

    semantic_values = np.array(
        [
            semantic_map[c]
            for c in common_careers
        ]
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    xgb_norm = minmax_normalize(
        xgb_values
    )

    semantic_norm = minmax_normalize(
        semantic_values
    )

    hybrid_scores = (
        alpha * xgb_norm
        + (1.0 - alpha) * semantic_norm
    )

    indices = np.argsort(
        hybrid_scores
    )[::-1][:TOP_K]

    rows = []

    print()
    print(
        f"{'Rank':<6}"
        f"{'Career':<55}"
        f"{'XGB':>10}"
        f"{'Semantic':>12}"
        f"{'Hybrid':>10}"
    )

    print("-" * 95)

    for rank, idx in enumerate(
        indices,
        start=1
    ):

        career = common_careers[idx]

        row = {
            "Rank": rank,
            "Career": career,
            "XGB_Probability": xgb_values[idx],
            "Semantic_Similarity": semantic_values[idx],
            "Hybrid_Score": hybrid_scores[idx],
        }

        rows.append(row)

        print(
            f"{rank:<6}"
            f"{career:<55}"
            f"{xgb_values[idx]:>10.6f}"
            f"{semantic_values[idx]:>12.6f}"
            f"{hybrid_scores[idx]:>10.6f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks(
    career_embeddings,
    career_labels,
    xgb_model,
    xgb_label_encoder
):

    section("SANITY CHECKS")

    print(
        "Semantic career embeddings shape:",
        career_embeddings.shape
    )

    print(
        "Semantic career labels:",
        len(career_labels)
    )

    print(
        "XGBoost classes:",
        len(xgb_model.classes_)
    )

    print(
        "XGBoost label encoder classes:",
        len(xgb_label_encoder.classes_)
    )

    print(
        "Expected careers:",
        878
    )

    # --------------------------------------------------------
    # Embedding norm check
    # --------------------------------------------------------

    norms = np.linalg.norm(
        career_embeddings,
        axis=1
    )

    print()
    print(
        "Career embedding norm:"
    )

    print(
        "  Minimum:",
        round(float(norms.min()), 6)
    )

    print(
        "  Maximum:",
        round(float(norms.max()), 6)
    )

    print(
        "  Mean   :",
        round(float(norms.mean()), 6)
    )

    if (
        abs(norms.mean() - 1.0)
        < 0.01
    ):
        print(
            "  STATUS : PASS"
        )
    else:
        print(
            "  STATUS : WARNING"
        )

    # --------------------------------------------------------
    # Duplicate career labels
    # --------------------------------------------------------

    duplicate_labels = (
        career_labels["Career"]
        .duplicated()
        .sum()
    )

    print()
    print(
        "Duplicate semantic career labels:",
        duplicate_labels
    )

    if duplicate_labels == 0:
        print(
            "Career label uniqueness: PASS"
        )
    else:
        print(
            "Career label uniqueness: WARNING"
        )

    # --------------------------------------------------------
    # Label intersection
    # --------------------------------------------------------

    try:

        xgb_careers = set(
            xgb_label_encoder.classes_
        )

        semantic_careers = set(
            career_labels["Career"]
            .astype(str)
        )

        intersection = (
            xgb_careers
            & semantic_careers
        )

        print()
        print(
            "XGBoost/SBERT common careers:",
            len(intersection)
        )

        if len(intersection) == 878:
            print(
                "Career alignment: PASS"
            )
        else:
            print(
                "Career alignment: WARNING"
            )

    except Exception as e:

        print(
            "Career alignment check failed:",
            e
        )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "CAREERCAST MILESTONE 2"
    )

    print(
        "READ-ONLY RECOMMENDATION DIAGNOSTIC"
    )

    print()
    print(
        "Existing model files will NOT be modified."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        tfidf,
        svd,
        riasec_scaler,
        tree_label_encoder,
        xgb_model,
        xgb_label_encoder,
        career_embeddings,
        career_labels,
        semantic_metadata,
    ) = load_artifacts()

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    run_sanity_checks(
        career_embeddings,
        career_labels,
        xgb_model,
        xgb_label_encoder,
    )

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    candidate = get_candidate()

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    X = build_xgb_features(
        candidate,
        tfidf,
        svd,
        riasec_scaler
    )

    (
        probabilities,
        model_careers,
        xgb_result,
    ) = run_xgb(
        xgb_model,
        xgb_label_encoder,
        career_labels,
        X,
    )

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    (
        semantic_similarities,
        semantic_result,
        semantic_model,
    ) = run_semantic(
        candidate,
        career_embeddings,
        career_labels,
    )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_result = run_hybrid(
        probabilities,
        model_careers,
        semantic_similarities,
        career_labels,
        ALPHA,
    )

    # --------------------------------------------------------
    # Final diagnosis
    # --------------------------------------------------------

    section("FINAL DIAGNOSIS")

    xgb_top1 = (
        xgb_result.iloc[0]["Career"]
    )

    semantic_top1 = (
        semantic_result.iloc[0]["Career"]
    )

    hybrid_top1 = (
        hybrid_result.iloc[0]["Career"]
    )

    print(
        "XGBoost Top-1 :",
        xgb_top1
    )

    print(
        "SBERT Top-1   :",
        semantic_top1
    )

    print(
        "Hybrid Top-1  :",
        hybrid_top1
    )

    print()
    print(
        "Alpha used:",
        ALPHA
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This script has not modified or retrained"
    )

    print(
        "any existing CareerCast model."
    )

    print()
    print(
        "Diagnostic complete."
    )


if __name__ == "__main__":
    main()
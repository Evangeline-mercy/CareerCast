# -*- coding: utf-8 -*-

"""
CareerCast Milestone 2
Corrected Final Career Recommendation Pipeline

IMPORTANT:
- Existing XGBoost model is NOT retrained.
- Existing artifacts are NOT modified.
- Education and Experience are converted to the numeric representation
  expected by the trained XGBoost model.
"""

import os
import re
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

HYBRID_METADATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "hybrid_recommender",
    "hybrid_recommender_metadata.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "corrected_final_recommendations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_DEFAULT = 0.60

TOP_K = 10


# ============================================================
# LOGGING
# ============================================================

def log(message):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}")


# ============================================================
# FILE CHECK
# ============================================================

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


def check_all_files():

    paths = [
        DATASET_PATH,
        TFIDF_PATH,
        SVD_PATH,
        RIASEC_SCALER_PATH,
        LABEL_ENCODER_PATH,
        XGB_MODEL_PATH,
        CAREER_EMBEDDINGS_PATH,
        CAREER_LABELS_PATH,
    ]

    for path in paths:
        check_file(path)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():

    log("Loading existing CareerCast artifacts...")

    dataset = pd.read_csv(DATASET_PATH)

    tfidf = joblib.load(TFIDF_PATH)
    svd = joblib.load(SVD_PATH)
    riasec_scaler = joblib.load(RIASEC_SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)

    career_embeddings = np.load(
        CAREER_EMBEDDINGS_PATH
    )

    career_df = pd.read_csv(
        CAREER_LABELS_PATH
    )

    log(f"Dataset rows       : {len(dataset)}")
    log(f"TF-IDF vocabulary  : {len(tfidf.vocabulary_)}")
    log(f"SVD components     : {svd.n_components}")
    log(f"Careers            : {len(label_encoder.classes_)}")
    log(f"Career embeddings  : {career_embeddings.shape}")

    return {
        "dataset": dataset,
        "tfidf": tfidf,
        "svd": svd,
        "riasec_scaler": riasec_scaler,
        "label_encoder": label_encoder,
        "xgb_model": xgb_model,
        "career_embeddings": career_embeddings,
        "career_df": career_df,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_artifacts(a):

    print()
    print("=" * 75)
    print("ARTIFACT VALIDATION")
    print("=" * 75)

    dataset = a["dataset"]
    tfidf = a["tfidf"]
    svd = a["svd"]
    label_encoder = a["label_encoder"]
    xgb_model = a["xgb_model"]
    career_embeddings = a["career_embeddings"]
    career_df = a["career_df"]

    required_columns = [
        "Profile_ID",
        "Career",
        "Skills",
        "Education",
        "Experience",
        "Realistic",
        "Investigative",
        "Artistic",
        "Social",
        "Enterprising",
        "Conventional",
        "Job_Description",
    ]

    missing = [
        c for c in required_columns
        if c not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset missing columns: {missing}"
        )

    if career_embeddings.shape[0] != len(career_df):
        raise ValueError(
            "Career embedding count does not match career labels."
        )

    if len(label_encoder.classes_) != len(career_df):
        raise ValueError(
            "Label encoder career count does not match career labels."
        )

    if xgb_model.n_classes_ != len(label_encoder.classes_):
        raise ValueError(
            "XGBoost class count does not match label encoder."
        )

    print("Dataset structure       : PASS")
    print("TF-IDF artifact         : PASS")
    print("SVD artifact            : PASS")
    print("RIASEC scaler           : PASS")
    print("Label encoder           : PASS")
    print("XGBoost model           : PASS")
    print("Career embeddings       : PASS")
    print("Career labels           : PASS")

    print()
    print("Artifact validation successful.")


# ============================================================
# EDUCATION CONVERSION
# ============================================================

def convert_education_to_training_scale(education_text, dataset):

    """
    The training dataset contains Education values approximately
    from 7.75 to 8.93.

    The exact training representation is numeric.

    For candidate inference, this function maps common degree
    descriptions to a representative value inside the observed
    training distribution.

    This is an inference compatibility layer only.
    """

    text = str(education_text).lower().strip()

    # If user directly enters a numeric value, preserve it.
    try:
        numeric = float(text)

        minimum = float(dataset["Education"].min())
        maximum = float(dataset["Education"].max())

        if minimum <= numeric <= maximum:
            return numeric

    except Exception:
        pass

    # Common degree mappings.
    #
    # These values deliberately stay within the training range.
    # For a B.E./B.Tech candidate, use a value near the upper
    # part of the observed education distribution.
    mappings = [

        (
            [
                "phd",
                "doctorate",
                "doctoral"
            ],
            8.90
        ),

        (
            [
                "m.tech",
                "m.e.",
                "master of engineering",
                "master of technology",
                "m.sc",
                "msc",
                "master"
            ],
            8.75
        ),

        (
            [
                "b.e",
                "b.e.",
                "be ",
                "b.tech",
                "btech",
                "b.sc",
                "bsc",
                "bachelor",
                "engineering"
            ],
            8.50
        ),

        (
            [
                "diploma",
                "polytechnic"
            ],
            8.10
        ),

        (
            [
                "12th",
                "higher secondary",
                "hsc"
            ],
            7.90
        ),

        (
            [
                "10th",
                "sslc",
                "secondary"
            ],
            7.80
        ),
    ]

    for keywords, value in mappings:

        for keyword in keywords:

            if keyword in text:
                return value

    # Safe fallback:
    # use median of training distribution.
    return float(dataset["Education"].median())


# ============================================================
# EXPERIENCE CONVERSION
# ============================================================

def convert_experience_to_years(experience_text):

    """
    Convert natural-language experience into numeric years.

    Examples:
        "6 months" -> 0.5
        "6 months internship experience" -> 0.5
        "1 year" -> 1.0
        "2 years" -> 2.0
    """

    text = str(experience_text).lower().strip()

    # Direct numeric input
    try:
        value = float(text)

        if value >= 0:
            return value

    except Exception:
        pass

    # Months
    month_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:months?|mos?)",
        text
    )

    if month_match:
        months = float(month_match.group(1))
        return months / 12.0

    # Years
    year_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
        text
    )

    if year_match:
        return float(year_match.group(1))

    # Decimal number anywhere in text
    number_match = re.search(
        r"(\d+(?:\.\d+)?)",
        text
    )

    if number_match:
        return float(number_match.group(1))

    return 0.0


# ============================================================
# RIASEC
# ============================================================

RIASEC_COLUMNS = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]


def scale_riasec(values, scaler):

    arr = np.array(
        [[
            values["Realistic"],
            values["Investigative"],
            values["Artistic"],
            values["Social"],
            values["Enterprising"],
            values["Conventional"],
        ]],
        dtype=float
    )

    return scaler.transform(arr)


# ============================================================
# BUILD XGBOOST FEATURES
# ============================================================

def build_xgb_features(
    skills,
    education_numeric,
    experience_numeric,
    riasec,
    artifacts
):

    tfidf = artifacts["tfidf"]
    svd = artifacts["svd"]
    scaler = artifacts["riasec_scaler"]

    # IMPORTANT:
    #
    # Training diagnostic showed that the original tree model
    # uses:
    #
    # TF-IDF/SVD over:
    # Skills + Education + Experience
    #
    # It does NOT use Job_Description.

    combined_text = (
        f"{skills} "
        f"{education_numeric} "
        f"{experience_numeric}"
    )

    tfidf_matrix = tfidf.transform(
        [combined_text]
    )

    svd_matrix = svd.transform(
        tfidf_matrix
    )

    riasec_scaled = scale_riasec(
        riasec,
        scaler
    )

    features = np.hstack(
        [
            svd_matrix,
            riasec_scaled
        ]
    )

    if features.shape[1] != 206:
        raise ValueError(
            f"Expected 206 XGBoost features, "
            f"got {features.shape[1]}"
        )

    return features


# ============================================================
# CAREER LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    return (
        str(label)
        .strip()
        .lower()
    )


def build_career_embedding_map(
    career_df,
    career_embeddings
):

    embedding_map = {}

    for idx, row in career_df.iterrows():

        label = str(
            row.iloc[0]
        ).strip()

        embedding_map[
            normalize_label(label)
        ] = career_embeddings[idx]

    return embedding_map


# ============================================================
# SEMANTIC PROFILE
# ============================================================

def build_semantic_text(
    skills,
    education,
    experience,
    job_description
):

    return (
        f"Skills: {skills}. "
        f"Education: {education}. "
        f"Experience: {experience}. "
        f"Job Description: {job_description}"
    )


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def minmax_normalize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum - minimum < 1e-12:

        return np.zeros_like(values)

    return (
        (values - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# SKILL EXTRACTION
# ============================================================

def tokenize_skills(text):

    text = str(text).lower()

    # Split common delimiters
    pieces = re.split(
        r"[,;|]",
        text
    )

    tokens = set()

    for piece in pieces:

        piece = piece.strip()

        if piece:
            tokens.add(piece)

    # Also capture important single-word technical terms.
    words = re.findall(
        r"[a-zA-Z0-9+#.]+",
        text
    )

    for word in words:

        word = word.lower().strip()

        if len(word) >= 2:
            tokens.add(word)

    return tokens


# ============================================================
# DATASET SKILL MAP
# ============================================================

def build_career_skill_map(dataset):

    career_skills = {}

    for career, group in dataset.groupby("Career"):

        skills = set()

        for value in group["Skills"].fillna(""):

            skills.update(
                tokenize_skills(value)
            )

        career_skills[
            str(career).strip()
        ] = skills

    return career_skills


# ============================================================
# RECOMMENDATION REASONS
# ============================================================

def get_skill_information(
    candidate_skills,
    career,
    career_skill_map
):

    candidate_tokens = tokenize_skills(
        candidate_skills
    )

    career_tokens = career_skill_map.get(
        career,
        set()
    )

    matched = sorted(
        candidate_tokens.intersection(
            career_tokens
        )
    )

    missing = sorted(
        career_tokens.difference(
            candidate_tokens
        )
    )

    # Limit gap list to 10
    missing = missing[:10]

    return matched, missing


# ============================================================
# XGBOOST PREDICTION
# ============================================================

def run_xgb_prediction(
    features,
    artifacts
):

    model = artifacts["xgb_model"]
    encoder = artifacts["label_encoder"]

    probabilities = model.predict_proba(
        features
    )[0]

    classes = encoder.classes_

    if len(probabilities) != len(classes):
        raise ValueError(
            "XGBoost probability count does not "
            "match label encoder classes."
        )

    return {
        str(classes[i]): float(probabilities[i])
        for i in range(len(classes))
    }


# ============================================================
# SBERT PREDICTION
# ============================================================

def run_semantic_prediction(
    semantic_text,
    artifacts,
    model
):

    career_embeddings = artifacts[
        "career_embeddings"
    ]

    career_df = artifacts[
        "career_df"
    ]

    candidate_embedding = model.encode(
        [semantic_text],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarities = cosine_similarity(
        candidate_embedding,
        career_embeddings
    )[0]

    labels = [
        str(x).strip()
        for x in career_df.iloc[:, 0]
    ]

    semantic_scores = {
        labels[i]: float(similarities[i])
        for i in range(len(labels))
    }

    return semantic_scores


# ============================================================
# HYBRID RANKING
# ============================================================

def build_hybrid_ranking(
    xgb_scores,
    semantic_scores,
    alpha
):

    common = set(
        xgb_scores.keys()
    ).intersection(
        semantic_scores.keys()
    )

    if len(common) != 878:
        print(
            f"WARNING: common career count = {len(common)}"
        )

    careers = sorted(common)

    xgb_values = np.array(
        [
            xgb_scores[c]
            for c in careers
        ]
    )

    semantic_values = np.array(
        [
            semantic_scores[c]
            for c in careers
        ]
    )

    xgb_norm = minmax_normalize(
        xgb_values
    )

    semantic_norm = minmax_normalize(
        semantic_values
    )

    hybrid_values = (
        alpha * xgb_norm
        +
        (1.0 - alpha) * semantic_norm
    )

    rows = []

    for i, career in enumerate(careers):

        rows.append(
            {
                "Career": career,
                "XGBoost_Probability":
                    float(xgb_values[i]),
                "Semantic_Similarity":
                    float(semantic_values[i]),
                "XGBoost_Normalized":
                    float(xgb_norm[i]),
                "Semantic_Normalized":
                    float(semantic_norm[i]),
                "Hybrid_Score":
                    float(hybrid_values[i]),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "Hybrid_Score",
        ascending=False
    ).reset_index(drop=True)

    result["Rank"] = (
        np.arange(len(result)) + 1
    )

    return result


# ============================================================
# ALPHA LOADING
# ============================================================

def get_validated_alpha():

    alpha = ALPHA_DEFAULT

    if os.path.exists(
        HYBRID_METADATA_PATH
    ):

        try:

            with open(
                HYBRID_METADATA_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                metadata = json.load(f)

            possible_keys = [
                "best_alpha",
                "validated_alpha",
                "alpha",
                "selected_alpha"
            ]

            for key in possible_keys:

                if key in metadata:

                    value = float(
                        metadata[key]
                    )

                    if 0.0 <= value <= 1.0:

                        alpha = value
                        break

        except Exception:
            pass

    return alpha


# ============================================================
# PRINT TOP 10
# ============================================================

def print_recommendations(
    recommendations,
    candidate,
    alpha,
    career_skill_map
):

    print()
    print("=" * 75)
    print("CAREERCAST - CORRECTED AI CAREER RECOMMENDATION")
    print("=" * 75)

    print()
    print(
        f"Validated XGBoost weight (alpha): "
        f"{alpha:.2f}"
    )

    print()
    print("CANDIDATE")
    print("-" * 75)

    print(
        f"Skills     : {candidate['skills']}"
    )

    print(
        f"Education  : {candidate['education']}"
    )

    print(
        f"Education numeric representation: "
        f"{candidate['education_numeric']:.2f}"
    )

    print(
        f"Experience : {candidate['experience']}"
    )

    print(
        f"Experience numeric years: "
        f"{candidate['experience_numeric']:.2f}"
    )

    print(
        f"Job Desc.  : {candidate['job_description']}"
    )

    print()
    print("RIASEC")
    print(
        f"  Realistic      : "
        f"{candidate['riasec']['Realistic']:.1f}"
    )
    print(
        f"  Investigative  : "
        f"{candidate['riasec']['Investigative']:.1f}"
    )
    print(
        f"  Artistic       : "
        f"{candidate['riasec']['Artistic']:.1f}"
    )
    print(
        f"  Social         : "
        f"{candidate['riasec']['Social']:.1f}"
    )
    print(
        f"  Enterprising   : "
        f"{candidate['riasec']['Enterprising']:.1f}"
    )
    print(
        f"  Conventional   : "
        f"{candidate['riasec']['Conventional']:.1f}"
    )

    print()
    print("=" * 75)
    print("TOP 10 CAREER RECOMMENDATIONS")
    print("=" * 75)

    for _, row in recommendations.head(10).iterrows():

        career = row["Career"]

        matched, gaps = get_skill_information(
            candidate["skills"],
            career,
            career_skill_map
        )

        print()
        print(
            f"{int(row['Rank'])}. {career}"
        )

        print(
            f"   XGBoost Probability : "
            f"{row['XGBoost_Probability']:.6f}"
        )

        print(
            f"   Semantic Similarity : "
            f"{row['Semantic_Similarity']:.6f}"
        )

        print(
            f"   Hybrid Score        : "
            f"{row['Hybrid_Score']:.6f}"
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

        if gaps:

            print(
                "   Suggested Skill Gaps: "
                + ", ".join(gaps)
            )

        else:

            print(
                "   Suggested Skill Gaps: None identified"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    recommendations,
    candidate,
    alpha
):

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"recommendations_{timestamp}.csv"
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        f"recommendation_summary_{timestamp}.json"
    )

    recommendations.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    top10 = recommendations.head(
        TOP_K
    ).copy()

    summary = {
        "model": "CareerCast Hybrid Recommender",
        "xgboost_weight_alpha": float(alpha),
        "semantic_weight": float(1.0 - alpha),
        "candidate": {
            "skills": candidate["skills"],
            "education": candidate["education"],
            "education_numeric": float(
                candidate["education_numeric"]
            ),
            "experience": candidate["experience"],
            "experience_numeric": float(
                candidate["experience_numeric"]
            ),
            "job_description":
                candidate["job_description"],
            "riasec": candidate["riasec"],
        },
        "top_10": []
    }

    for _, row in top10.iterrows():

        summary["top_10"].append(
            {
                "rank": int(row["Rank"]),
                "career": row["Career"],
                "xgboost_probability":
                    float(row["XGBoost_Probability"]),
                "semantic_similarity":
                    float(row["Semantic_Similarity"]),
                "hybrid_score":
                    float(row["Hybrid_Score"]),
            }
        )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    return csv_path, json_path


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print()
    print("=" * 75)
    print("CAREERCAST MILESTONE 2")
    print("CORRECTED FINAL CAREER RECOMMENDATION PIPELINE")
    print("=" * 75)

    print()
    print("READ-ONLY MODEL MODE")
    print(
        "Existing XGBoost and SBERT artifacts "
        "will NOT be modified."
    )

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    print()
    print("Stage 1/7  Checking artifacts...")

    check_all_files()

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    print("Stage 2/7  Loading artifacts...")

    artifacts = load_artifacts()

    validate_artifacts(
        artifacts
    )

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------

    print()
    print("Stage 3/7  Loading Sentence-BERT...")

    print(
        "Model: all-MiniLM-L6-v2"
    )

    semantic_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "Sentence-BERT loaded successfully."
    )

    # --------------------------------------------------------
    # Stage 4
    # --------------------------------------------------------

    print()
    print("Stage 4/7  Candidate input")

    print()
    print(
        "Enter candidate information."
    )

    skills = input(
        "\nSkills: "
    ).strip()

    education = input(
        "Education: "
    ).strip()

    experience = input(
        "Experience: "
    ).strip()

    job_description = input(
        "Job Description: "
    ).strip()

    print()
    print(
        "Enter the six RIASEC scores."
    )

    print(
        "Use the same scale used in the training dataset."
    )

    riasec = {}

    for key in RIASEC_COLUMNS:

        while True:

            try:

                value = float(
                    input(
                        f"{key}: "
                    )
                )

                riasec[key] = value
                break

            except ValueError:

                print(
                    "Please enter a numeric value."
                )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    education_numeric = (
        convert_education_to_training_scale(
            education,
            artifacts["dataset"]
        )
    )

    experience_numeric = (
        convert_experience_to_years(
            experience
        )
    )

    candidate = {
        "skills": skills,
        "education": education,
        "education_numeric": education_numeric,
        "experience": experience,
        "experience_numeric": experience_numeric,
        "job_description": job_description,
        "riasec": riasec,
    }

    print()
    print("=" * 75)
    print("TRAINING-COMPATIBLE CANDIDATE REPRESENTATION")
    print("=" * 75)

    print(
        f"Education text : {education}"
    )

    print(
        f"Education used : {education_numeric:.4f}"
    )

    print(
        f"Experience text: {experience}"
    )

    print(
        f"Experience used: {experience_numeric:.4f} years"
    )

    # --------------------------------------------------------
    # Stage 5
    # --------------------------------------------------------

    print()
    print(
        "Stage 5/7  Building XGBoost features..."
    )

    features = build_xgb_features(
        skills=skills,
        education_numeric=education_numeric,
        experience_numeric=experience_numeric,
        riasec=riasec,
        artifacts=artifacts
    )

    print(
        f"XGBoost feature shape: {features.shape}"
    )

    # XGBoost prediction
    xgb_scores = run_xgb_prediction(
        features,
        artifacts
    )

    # --------------------------------------------------------
    # Stage 6
    # --------------------------------------------------------

    print()
    print(
        "Stage 6/7  Running semantic + hybrid inference..."
    )

    semantic_text = build_semantic_text(
        skills,
        education,
        experience,
        job_description
    )

    semantic_scores = run_semantic_prediction(
        semantic_text,
        artifacts,
        semantic_model
    )

    alpha = get_validated_alpha()

    print(
        f"Validated alpha: {alpha:.2f}"
    )

    recommendations = build_hybrid_ranking(
        xgb_scores,
        semantic_scores,
        alpha
    )

    # --------------------------------------------------------
    # Stage 7
    # --------------------------------------------------------

    print()
    print(
        "Stage 7/7  Generating recommendation report..."
    )

    career_skill_map = (
        build_career_skill_map(
            artifacts["dataset"]
        )
    )

    print_recommendations(
        recommendations,
        candidate,
        alpha,
        career_skill_map
    )

    csv_path, json_path = save_results(
        recommendations,
        candidate,
        alpha
    )

    elapsed = time.time() - total_start

    print()
    print("=" * 75)
    print("RECOMMENDATION COMPLETE")
    print("=" * 75)

    print()
    print(
        "Full ranking saved to:"
    )

    print(csv_path)

    print()
    print(
        "Summary saved to:"
    )

    print(json_path)

    print()
    print(
        f"Total time: {elapsed:.2f}s"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "- Existing XGBoost model was NOT retrained."
    )

    print(
        "- Existing model artifacts were NOT modified."
    )

    print(
        "- Existing recommendation files were NOT overwritten."
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
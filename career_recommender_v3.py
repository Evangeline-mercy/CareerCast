# ============================================================
# CAREERCAST MILESTONE 2
# V3 - CORRECTED HYBRID CAREER RECOMMENDER
#
# READ-ONLY:
#   - Does NOT retrain XGBoost
#   - Does NOT modify dataset
#   - Does NOT modify SBERT embeddings
#   - Does NOT modify existing models
#   - Does NOT overwrite existing recommendation files
#
# XGBoost features:
#   200 SVD features
#   + 6 RIASEC features
#   = 206 features
# ============================================================

import os
import re
import json
import math
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "results/careercast_candidate_profiles.csv"

# XGBoost feature artifacts
TFIDF_PATH = "results/tree_features/tfidf_vectorizer.joblib"
SVD_PATH = "results/tree_features/svd.joblib"
RIASEC_SCALER_PATH = "results/tree_features/riasec_scaler.joblib"

# XGBoost model
XGBOOST_PATH = "results/xgboost_model/xgboost_model.joblib"
XGBOOST_LABEL_ENCODER_PATH = (
    "results/xgboost_model/label_encoder.joblib"
)

# Existing fine-tuned SBERT career embeddings
CAREER_EMBEDDINGS_PATH = (
    "results/semantic_embeddings/finetuned_career_embeddings.npy"
)

CAREER_LABELS_PATH = (
    "results/semantic_embeddings/finetuned_career_labels.csv"
)

# Existing fine-tuned SBERT model
SBERT_MODEL_NAME = (
    "results/semantic_embeddings/sbert_finetuned"
)

# New output only
OUTPUT_ROOT = "v3_output"

# Hybrid weights
WEIGHT_SEMANTIC = 0.60
WEIGHT_SKILL = 0.30
WEIGHT_XGBOOST = 0.10

TOP_K = 15

EXPECTED_XGB_FEATURES = 206


# ============================================================
# CANDIDATE INPUT
# ============================================================

CANDIDATE = {

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

    "riasec": {
        "Realistic": 6,
        "Investigative": 9,
        "Artistic": 4,
        "Social": 5,
        "Enterprising": 5,
        "Conventional": 8,
    }
}


# ============================================================
# RIASEC ORDER
# ============================================================

RIASEC_ORDER = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional"
]


# ============================================================
# SKILL NORMALIZATION
# ============================================================

def normalize_skill(skill):

    if not isinstance(skill, str):
        return ""

    s = skill.lower().strip()

    aliases = {

        "structured query language sql":
            "sql",

        "structured query language":
            "sql",

        "python programming":
            "python",

        "python programming language":
            "python",

        "machine learning software":
            "machine learning",

        "machine learning technology":
            "machine learning",

        "numpy software":
            "numpy",

        "pandas software":
            "pandas",

        "tensorflow software":
            "tensorflow",

        "tensorflow framework":
            "tensorflow",

        "data analytics":
            "data analysis",

        "data analysis software":
            "data analysis",

        "artificial intelligence":
            "machine learning",

        "machine intelligence":
            "machine learning",
    }

    # Remove punctuation
    s = re.sub(
        r"[^a-z0-9\s]",
        " ",
        s
    )

    # Collapse spaces
    s = re.sub(
        r"\s+",
        " ",
        s
    ).strip()

    # Apply aliases
    if s in aliases:
        return aliases[s]

    return s


# ============================================================
# PARSE SKILL STRING
# ============================================================

def parse_skill_string(raw):

    """
    Convert comma-separated skills into
    normalized, duplicate-free set.
    """

    if not isinstance(raw, str):
        return set()

    if not raw.strip():
        return set()

    skills = set()

    for part in raw.split(","):

        normalized = normalize_skill(part)

        if normalized:
            skills.add(normalized)

    return skills


# ============================================================
# STEP 1
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 75)
    print("CAREERCAST MILESTONE 2")
    print("V3 CORRECTED HYBRID CAREER RECOMMENDER")
    print("=" * 75)

    print("\nREAD-ONLY MODE")
    print("Dataset will NOT be modified.")
    print("XGBoost will NOT be retrained.")
    print("SBERT embeddings will NOT be modified.")
    print("Existing results will NOT be overwritten.")

    print("\n" + "=" * 75)
    print("1. LOADING DATASET")
    print("=" * 75)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required = [
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
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Dataset rows :", len(df))
    print(
        "Unique careers:",
        df["Career"].nunique()
    )

    print("Required columns: PASS")

    return df


# ============================================================
# STEP 2
# BUILD CAREER-LEVEL SKILL SETS
# ============================================================

def build_career_skill_sets(df):

    print("\n" + "=" * 75)
    print("2. BUILDING CAREER-LEVEL SKILL SETS")
    print("=" * 75)

    career_skills = defaultdict(set)

    for _, row in df.iterrows():

        career = str(row["Career"]).strip()

        skills = parse_skill_string(
            row["Skills"]
        )

        career_skills[career].update(
            skills
        )

    career_skills = dict(career_skills)

    print(
        "Career skill sets:",
        len(career_skills)
    )

    return career_skills


# ============================================================
# STEP 3
# BUILD IDF WEIGHTS
# ============================================================

def build_skill_idf(career_skill_sets):

    print("\n" + "=" * 75)
    print("3. BUILDING IDF SKILL WEIGHTS")
    print("=" * 75)

    number_of_careers = len(
        career_skill_sets
    )

    if number_of_careers == 0:
        raise ValueError(
            "No careers found in dataset."
        )

    document_frequency = defaultdict(int)

    for skills in career_skill_sets.values():

        for skill in skills:
            document_frequency[skill] += 1

    idf = {}

    for skill, df in document_frequency.items():

        idf[skill] = math.log(
            number_of_careers /
            (1 + df)
        )

    print(
        "Unique skills:",
        len(idf)
    )

    return idf


# ============================================================
# STEP 4
# IMPROVED SKILL MATCHING
# ============================================================

def calculate_skill_match(
    candidate_skills,
    career_skills,
    idf
):

    """
    Improved career skill matching.

    Features:
    1. Exact normalized skill matching
    2. Alias-aware matching
    3. Token-based partial matching
    4. Weighted candidate skill coverage
    5. Simple skill coverage
    """

    if not candidate_skills:
        return 0.0, []

    if not career_skills:
        return 0.0, []

    matched = set()

    # --------------------------------------------------------
    # EXACT MATCHING
    # --------------------------------------------------------

    for candidate_skill in candidate_skills:

        if candidate_skill in career_skills:

            matched.add(
                candidate_skill
            )

    # --------------------------------------------------------
    # PARTIAL / TOKEN MATCHING
    # --------------------------------------------------------

    for candidate_skill in candidate_skills:

        if candidate_skill in matched:
            continue

        candidate_tokens = set(
            candidate_skill.split()
        )

        if not candidate_tokens:
            continue

        for career_skill in career_skills:

            career_tokens = set(
                career_skill.split()
            )

            overlap = (
                len(
                    candidate_tokens
                    &
                    career_tokens
                )
                /
                len(candidate_tokens)
            )

            if overlap >= 0.80:

                matched.add(
                    candidate_skill
                )

                break

    # --------------------------------------------------------
    # NO MATCH
    # --------------------------------------------------------

    if not matched:
        return 0.0, []

    # --------------------------------------------------------
    # WEIGHT EACH CANDIDATE SKILL
    # --------------------------------------------------------

    candidate_weights = {}

    for skill in candidate_skills:

        raw_idf = idf.get(
            skill,
            1.0
        )

        candidate_weights[skill] = max(
            float(raw_idf),
            0.10
        )

    # --------------------------------------------------------
    # TOTAL CANDIDATE WEIGHT
    # --------------------------------------------------------

    total_weight = sum(
        candidate_weights.values()
    )

    if total_weight <= 0:
        return 0.0, sorted(matched)

    # --------------------------------------------------------
    # MATCHED WEIGHT
    # --------------------------------------------------------

    matched_weight = sum(
        candidate_weights[skill]
        for skill in matched
    )

    # --------------------------------------------------------
    # WEIGHTED COVERAGE
    # --------------------------------------------------------

    weighted_coverage = (
        matched_weight /
        total_weight
    )

    # --------------------------------------------------------
    # SIMPLE COVERAGE
    # --------------------------------------------------------

    skill_coverage = (
        len(matched) /
        len(candidate_skills)
    )

    # --------------------------------------------------------
    # FINAL SKILL SCORE
    # --------------------------------------------------------

    score = (
        0.65 * weighted_coverage
        +
        0.35 * skill_coverage
    )

    score = max(
        0.0,
        min(1.0, score)
    )

    return score, sorted(matched)


# ============================================================
# STEP 5
# LOAD SBERT
# ============================================================

def load_sbert():

    print("\n" + "=" * 75)
    print("4. LOADING SENTENCE-BERT")
    print("=" * 75)

    from sentence_transformers import (
        SentenceTransformer
    )

    print(
        "Model:",
        SBERT_MODEL_NAME
    )

    if not os.path.exists(
        SBERT_MODEL_NAME
    ):
        raise FileNotFoundError(
            f"SBERT model not found:\n"
            f"{SBERT_MODEL_NAME}"
        )

    model = SentenceTransformer(
        SBERT_MODEL_NAME
    )

    print(
        "Sentence-BERT loaded successfully."
    )

    return model


# ============================================================
# STEP 6
# LOAD CAREER EMBEDDINGS
# ============================================================

def load_career_embeddings():

    print("\n" + "=" * 75)
    print("5. LOADING CAREER EMBEDDINGS")
    print("=" * 75)

    if not os.path.exists(
        CAREER_EMBEDDINGS_PATH
    ):
        raise FileNotFoundError(
            "Career embeddings not found:\n"
            f"{CAREER_EMBEDDINGS_PATH}"
        )

    if not os.path.exists(
        CAREER_LABELS_PATH
    ):
        raise FileNotFoundError(
            "Career labels not found:\n"
            f"{CAREER_LABELS_PATH}"
        )

    embeddings = np.load(
        CAREER_EMBEDDINGS_PATH
    )

    labels_df = pd.read_csv(
        CAREER_LABELS_PATH
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Career labels:",
        len(labels_df)
    )

    # --------------------------------------------------------
    # Detect career-label column
    # --------------------------------------------------------

    possible_columns = [
        "Career",
        "career",
        "Label",
        "label"
    ]

    label_column = None

    for column in possible_columns:

        if column in labels_df.columns:

            label_column = column
            break

    if label_column is None:

        if len(labels_df.columns) == 1:

            label_column = (
                labels_df.columns[0]
            )

        else:

            raise ValueError(
                "Could not identify career "
                "label column in career labels CSV."
            )

    career_order = (
        labels_df[label_column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    # --------------------------------------------------------
    # HARD ALIGNMENT CHECK
    # --------------------------------------------------------

    if len(career_order) != embeddings.shape[0]:

        raise ValueError(
            "\nCAREER ALIGNMENT ERROR\n"
            "Number of career labels does not "
            "match number of embedding rows.\n"
            f"Embeddings: {embeddings.shape[0]}\n"
            f"Labels:     {len(career_order)}"
        )

    # --------------------------------------------------------
    # Check duplicate career labels
    # --------------------------------------------------------

    duplicate_labels = (
        pd.Series(career_order)
        .duplicated()
    )

    if duplicate_labels.any():

        duplicates = (
            pd.Series(career_order)[
                duplicate_labels
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate career labels found "
            "in career labels file:\n"
            f"{duplicates[:10]}"
        )

    print(
        "Career label column:",
        label_column
    )

    print(
        "Career order verified: PASS"
    )

    return embeddings, career_order


# ============================================================
# STEP 7
# COMPUTE SBERT SEMANTIC SIMILARITY
# ============================================================

def compute_semantic_scores(
    candidate,
    career_embeddings,
    career_order,
    sbert_model
):

    print("\n" + "=" * 75)
    print("6. COMPUTING SBERT SEMANTIC SIMILARITY")
    print("=" * 75)

    candidate_text = (
        f"Skills: {candidate['skills']}. "
        f"Education: {candidate['education']}. "
        f"Experience: {candidate['experience']}. "
        f"Job Description: "
        f"{candidate['job_description']}"
    )

    print("\nCandidate semantic text:")
    print(candidate_text)

    candidate_embedding = (
        sbert_model.encode(
            [candidate_text],
            normalize_embeddings=True
        )
    )

    # --------------------------------------------------------
    # Normalize existing embeddings safely
    # --------------------------------------------------------

    career_norm = (
        career_embeddings /
        (
            np.linalg.norm(
                career_embeddings,
                axis=1,
                keepdims=True
            )
            + 1e-12
        )
    )

    similarities = (
        career_norm
        @
        candidate_embedding[0]
    )

    # --------------------------------------------------------
    # Alignment check
    # --------------------------------------------------------

    if len(similarities) != len(
        career_order
    ):

        raise ValueError(
            "SBERT alignment error: "
            "similarity count does not match "
            "career label count."
        )

    print(
        "Similarity count:",
        len(similarities)
    )

    print(
        "Minimum:",
        round(
            float(similarities.min()),
            6
        )
    )

    print(
        "Maximum:",
        round(
            float(similarities.max()),
            6
        )
    )

    print(
        "SBERT career alignment: PASS"
    )

    return similarities


# ============================================================
# STEP 8
# BUILD EXACT 206-FEATURE XGBOOST VECTOR
# ============================================================

def build_xgboost_features(
    candidate,
    tfidf,
    svd,
    riasec_scaler
):

    print("\n" + "=" * 75)
    print("7. BUILDING XGBOOST FEATURES")
    print("=" * 75)

    # --------------------------------------------------------
    # Existing XGBoost structure:
    #
    # 200 SVD features
    # + 6 RIASEC features
    # = 206 features
    #
    # --------------------------------------------------------

    combined_text = (
        f"{candidate['skills']} "
        f"{candidate['education']} "
        f"{candidate['experience']} "
        f"{candidate['job_description']}"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf_vector = tfidf.transform(
        [combined_text]
    )

    print(
        "TF-IDF shape:",
        tfidf_vector.shape
    )

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    svd_vector = svd.transform(
        tfidf_vector
    )

    print(
        "SVD shape:",
        svd_vector.shape
    )

    if svd_vector.shape[1] != 200:

        raise ValueError(
            "SVD feature mismatch!\n"
            f"Expected 200 features, "
            f"got {svd_vector.shape[1]}"
        )

    # --------------------------------------------------------
    # RIASEC
    # --------------------------------------------------------

    missing_riasec = [
        key
        for key in RIASEC_ORDER
        if key not in candidate["riasec"]
    ]

    if missing_riasec:

        raise ValueError(
            "Missing RIASEC values:\n"
            f"{missing_riasec}"
        )

    riasec_values = np.array(
        [[
            candidate["riasec"][key]
            for key in RIASEC_ORDER
        ]],
        dtype=np.float32
    )

    riasec_scaled = (
        riasec_scaler.transform(
            riasec_values
        )
    )

    print(
        "RIASEC shape:",
        riasec_scaled.shape
    )

    if riasec_scaled.shape[1] != 6:

        raise ValueError(
            "RIASEC feature mismatch!\n"
            f"Expected 6 features, "
            f"got {riasec_scaled.shape[1]}"
        )

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = np.hstack(
        [
            svd_vector,
            riasec_scaled
        ]
    )

    print(
        "Final XGBoost feature shape:",
        features.shape
    )

    # --------------------------------------------------------
    # HARD SAFETY CHECK
    # --------------------------------------------------------

    if features.shape[1] != (
        EXPECTED_XGB_FEATURES
    ):

        raise ValueError(
            "XGBoost feature mismatch!\n"
            f"Expected "
            f"{EXPECTED_XGB_FEATURES}, "
            f"got {features.shape[1]}"
        )

    print(
        "206-feature verification: PASS"
    )

    return features


# ============================================================
# STEP 9
# XGBOOST INFERENCE + CAREER ALIGNMENT
# ============================================================

def compute_xgboost_scores(
    features,
    xgb_model,
    label_encoder,
    career_order
):

    print("\n" + "=" * 75)
    print("8. XGBOOST INFERENCE")
    print("=" * 75)

    probabilities = (
        xgb_model.predict_proba(
            features
        )
    )

    # --------------------------------------------------------
    # Check probability matrix
    # --------------------------------------------------------

    if probabilities.ndim != 2:

        raise ValueError(
            "Unexpected XGBoost probability "
            "shape:\n"
            f"{probabilities.shape}"
        )

    if probabilities.shape[0] != 1:

        raise ValueError(
            "Expected one candidate row "
            "from XGBoost."
        )

    probabilities = probabilities[0]

    print(
        "Probability count:",
        len(probabilities)
    )

    # --------------------------------------------------------
    # Check model / encoder class count
    # --------------------------------------------------------

    if hasattr(
        xgb_model,
        "classes_"
    ):

        model_classes = (
            np.asarray(
                xgb_model.classes_
            )
        )

    else:

        model_classes = np.arange(
            len(probabilities)
        )

    if len(model_classes) != len(
        probabilities
    ):

        raise ValueError(
            "XGBoost class/probability "
            "count mismatch.\n"
            f"Classes:       "
            f"{len(model_classes)}\n"
            f"Probabilities: "
            f"{len(probabilities)}"
        )

    if len(label_encoder.classes_) != len(
        probabilities
    ):

        raise ValueError(
            "Label encoder / XGBoost "
            "probability count mismatch.\n"
            f"Label encoder classes: "
            f"{len(label_encoder.classes_)}\n"
            f"Probabilities: "
            f"{len(probabilities)}"
        )

    # --------------------------------------------------------
    # Convert encoded classes to career names
    # --------------------------------------------------------

    try:

        class_names = (
            label_encoder.inverse_transform(
                model_classes.astype(int)
            )
        )

    except Exception as exc:

        raise ValueError(
            "Could not align XGBoost "
            "classes with label encoder."
        ) from exc

    class_names = [
        str(name).strip()
        for name in class_names
    ]

    # --------------------------------------------------------
    # Check duplicate class names
    # --------------------------------------------------------

    if len(class_names) != len(
        set(class_names)
    ):

        raise ValueError(
            "Duplicate career names found "
            "after XGBoost label decoding."
        )

    # --------------------------------------------------------
    # Build probability map
    # --------------------------------------------------------

    probability_by_career = dict(
        zip(
            class_names,
            probabilities
        )
    )

    # --------------------------------------------------------
    # ALIGN XGBoost RESULTS TO SBERT
    # CAREER ORDER
    # --------------------------------------------------------

    aligned_scores = np.array(
        [
            probability_by_career.get(
                career,
                0.0
            )
            for career in career_order
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Count missing careers
    # --------------------------------------------------------

    missing_careers = [
        career
        for career in career_order
        if career not in probability_by_career
    ]

    if missing_careers:

        print(
            "\nWARNING:"
        )

        print(
            "Careers in SBERT order but "
            "not in XGBoost classes:",
            len(missing_careers)
        )

        print(
            "First missing careers:",
            missing_careers[:10]
        )

    # --------------------------------------------------------
    # HARD ALIGNMENT CHECK
    # --------------------------------------------------------

    if len(aligned_scores) != len(
        career_order
    ):

        raise ValueError(
            "XGBoost alignment failed."
        )

    print(
        "XGBoost class alignment: PASS"
    )

    print(
        "Aligned XGBoost scores:",
        len(aligned_scores)
    )

    return aligned_scores


# ============================================================
# STEP 10
# NORMALIZATION
# ============================================================

def min_max_normalize(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    minimum = values.min()
    maximum = values.max()

    if (
        maximum - minimum
        < 1e-12
    ):

        return np.zeros_like(
            values
        )

    return (
        (values - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# STEP 11
# HYBRID RANKING
# ============================================================

def calculate_final_scores(
    semantic_scores,
    skill_scores,
    xgb_scores
):

    print("\n" + "=" * 75)
    print("9. HYBRID SCORING")
    print("=" * 75)

    # --------------------------------------------------------
    # HARD ARRAY ALIGNMENT CHECK
    # --------------------------------------------------------

    lengths = {
        "semantic": len(
            semantic_scores
        ),
        "skill": len(
            skill_scores
        ),
        "xgboost": len(
            xgb_scores
        )
    }

    if len(
        set(lengths.values())
    ) != 1:

        raise ValueError(
            "Hybrid score alignment error.\n"
            f"Score lengths: {lengths}"
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    semantic_normalized = (
        min_max_normalize(
            semantic_scores
        )
    )

    skill_normalized = (
        min_max_normalize(
            skill_scores
        )
    )

    xgb_normalized = (
        min_max_normalize(
            xgb_scores
        )
    )

    # --------------------------------------------------------
    # Hybrid formula
    # --------------------------------------------------------

    final_scores = (

        WEIGHT_SEMANTIC
        * semantic_normalized

        +

        WEIGHT_SKILL
        * skill_normalized

        +

        WEIGHT_XGBOOST
        * xgb_normalized
    )

    print(
        f"Semantic weight : "
        f"{WEIGHT_SEMANTIC}"
    )

    print(
        f"Skill weight    : "
        f"{WEIGHT_SKILL}"
    )

    print(
        f"XGBoost weight  : "
        f"{WEIGHT_XGBOOST}"
    )

    print(
        "Weight sum      :",
        WEIGHT_SEMANTIC
        + WEIGHT_SKILL
        + WEIGHT_XGBOOST
    )

    return (
        final_scores,
        semantic_normalized,
        skill_normalized,
        xgb_normalized
    )


# ============================================================
# STEP 12
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence_scores(
    final_scores
):

    """
    Convert hybrid ranking scores into
    softmax-based relative confidence scores.

    NOTE:
    This is a ranking confidence indicator,
    NOT a calibrated probability of employment.
    """

    scores = np.asarray(
        final_scores,
        dtype=np.float64
    )

    if len(scores) == 0:

        return np.array(
            [],
            dtype=np.float64
        )

    # Numerical stability
    shifted_scores = (
        scores - np.max(scores)
    )

    exp_scores = np.exp(
        shifted_scores
    )

    total = np.sum(
        exp_scores
    )

    if total <= 0:

        return np.zeros_like(
            scores
        )

    probabilities = (
        exp_scores /
        total
    )

    return probabilities


# ============================================================
# STEP 13
# SKILL ALIGNMENT METRICS
# ============================================================

def calculate_skill_alignment(
    candidate_skills,
    matched_skills
):

    total_skills = len(
        candidate_skills
    )

    matched_count = len(
        matched_skills
    )

    if total_skills == 0:

        return {
            "total_candidate_skills": 0,
            "matched_skill_count": 0,
            "skill_alignment_percentage": 0.0
        }

    alignment_percentage = (
        matched_count /
        total_skills
    ) * 100

    return {

        "total_candidate_skills":
            total_skills,

        "matched_skill_count":
            matched_count,

        "skill_alignment_percentage":
            round(
                alignment_percentage,
                2
            )
    }


# ============================================================
# STEP 14
# DISPLAY RESULTS
# ============================================================

def display_results(results):

    print("\n" + "=" * 75)
    print(
        f"TOP {TOP_K} CAREER RECOMMENDATIONS"
    )
    print("=" * 75)

    for result in results:

        matched = (

            ", ".join(
                result["matched_skills"]
            )

            if result["matched_skills"]

            else "None"
        )

        print(
            f"\n{result['rank']}. "
            f"{result['career']}"
        )

        print(
            f"   FINAL SCORE          : "
            f"{result['final_score']:.4f}"
        )

        print(
            f"   CONFIDENCE           : "
            f"{result['confidence_percentage']:.2f}%"
        )

        print(
            f"   Semantic Score       : "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"   Skill Match          : "
            f"{result['skill_match']:.4f}"
        )

        print(
            f"   XGBoost Probability  : "
            f"{result['xgboost_probability']:.6f}"
        )

        print(
            f"   Matched Skills       : "
            f"{matched}"
        )

        alignment = (
            result["skill_alignment"]
        )

        print(
            f"   Skill Alignment      : "
            f"{alignment['skill_alignment_percentage']:.2f}%"
        )

    print("\n" + "=" * 75)


# ============================================================
# STEP 15
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    candidate
):

    print("\n" + "=" * 75)
    print("11. SAVING RESULTS")
    print("=" * 75)

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_directory = os.path.join(
        OUTPUT_ROOT,
        f"run_{timestamp}"
    )

    os.makedirs(
        output_directory
    )

    json_path = os.path.join(
        output_directory,
        "v3_recommendations.json"
    )

    csv_path = os.path.join(
        output_directory,
        "v3_recommendations.csv"
    )

    output_data = {

        "version": "V3",

        "weights": {

            "semantic":
                WEIGHT_SEMANTIC,

            "skill":
                WEIGHT_SKILL,

            "xgboost":
                WEIGHT_XGBOOST
        },

        "xgboost_features": {

            "svd_features": 200,

            "riasec_features": 6,

            "total_features": 206
        },

        "candidate": candidate,

        "results": results
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    pd.DataFrame(
        results
    ).to_csv(
        csv_path,
        index=False
    )

    print(
        "JSON saved:",
        json_path
    )

    print(
        "CSV saved:",
        csv_path
    )

    return output_directory


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = load_dataset()

    # ========================================================
    # BUILD CAREER SKILL SETS
    # ========================================================

    career_skill_sets = (
        build_career_skill_sets(
            df
        )
    )

    # ========================================================
    # BUILD IDF
    # ========================================================

    idf = build_skill_idf(
        career_skill_sets
    )

    # ========================================================
    # CANDIDATE SKILLS
    # ========================================================

    candidate_skills = (
        parse_skill_string(
            CANDIDATE["skills"]
        )
    )

    print(
        "\nCandidate normalized skills:"
    )

    print(
        sorted(candidate_skills)
    )

    # ========================================================
    # LOAD CAREER EMBEDDINGS
    # ========================================================

    career_embeddings, career_order = (
        load_career_embeddings()
    )

    # ========================================================
    # DATASET CAREER ALIGNMENT
    # ========================================================

    dataset_careers = set(
        career_skill_sets.keys()
    )

    missing_from_dataset = [

        career

        for career in career_order

        if career not in dataset_careers
    ]

    if missing_from_dataset:

        print(
            "\nWARNING:"
        )

        print(
            "Careers in embedding labels "
            "but not found in dataset:",
            len(
                missing_from_dataset
            )
        )

        print(
            "First missing careers:",
            missing_from_dataset[:10]
        )

    else:

        print(
            "\nDataset / SBERT career alignment: PASS"
        )

    # ========================================================
    # CALCULATE SKILL SCORES
    # ========================================================

    skill_scores = []

    matched_skills_by_career = {}

    for career in career_order:

        career_skills = (
            career_skill_sets.get(
                career,
                set()
            )
        )

        score, matched = (
            calculate_skill_match(
                candidate_skills,
                career_skills,
                idf
            )
        )

        skill_scores.append(
            score
        )

        matched_skills_by_career[
            career
        ] = matched

    skill_scores = np.array(
        skill_scores,
        dtype=np.float64
    )

    # ========================================================
    # LOAD SBERT
    # ========================================================

    sbert_model = load_sbert()

    # ========================================================
    # SEMANTIC SCORES
    # ========================================================

    semantic_scores = (
        compute_semantic_scores(
            CANDIDATE,
            career_embeddings,
            career_order,
            sbert_model
        )
    )

    # ========================================================
    # LOAD XGBOOST ARTIFACTS
    # ========================================================

    print("\n" + "=" * 75)
    print("7. LOADING XGBOOST ARTIFACTS")
    print("=" * 75)

    if not os.path.exists(
        TFIDF_PATH
    ):
        raise FileNotFoundError(
            f"TF-IDF artifact not found:\n"
            f"{TFIDF_PATH}"
        )

    if not os.path.exists(
        SVD_PATH
    ):
        raise FileNotFoundError(
            f"SVD artifact not found:\n"
            f"{SVD_PATH}"
        )

    if not os.path.exists(
        RIASEC_SCALER_PATH
    ):
        raise FileNotFoundError(
            f"RIASEC scaler not found:\n"
            f"{RIASEC_SCALER_PATH}"
        )

    if not os.path.exists(
        XGBOOST_PATH
    ):
        raise FileNotFoundError(
            f"XGBoost model not found:\n"
            f"{XGBOOST_PATH}"
        )

    if not os.path.exists(
        XGBOOST_LABEL_ENCODER_PATH
    ):
        raise FileNotFoundError(
            f"XGBoost label encoder not found:\n"
            f"{XGBOOST_LABEL_ENCODER_PATH}"
        )

    tfidf = joblib.load(
        TFIDF_PATH
    )

    svd = joblib.load(
        SVD_PATH
    )

    riasec_scaler = joblib.load(
        RIASEC_SCALER_PATH
    )

    xgb_model = joblib.load(
        XGBOOST_PATH
    )

    label_encoder = joblib.load(
        XGBOOST_LABEL_ENCODER_PATH
    )

    print(
        "TF-IDF vocabulary:",
        len(
            tfidf.vocabulary_
        )
    )

    print(
        "SVD components:",
        svd.n_components
    )

    print(
        "RIASEC features:",
        riasec_scaler.n_features_in_
    )

    print(
        "XGBoost expected features:",
        xgb_model.n_features_in_
    )

    print(
        "XGBoost classes:",
        len(
            label_encoder.classes_
        )
    )

    # ========================================================
    # VERIFY XGBOOST MODEL FEATURE COUNT
    # ========================================================

    if xgb_model.n_features_in_ != (
        EXPECTED_XGB_FEATURES
    ):

        raise ValueError(
            "Existing XGBoost model does "
            "not expect 206 features.\n"
            f"Model expects: "
            f"{xgb_model.n_features_in_}\n"
            f"Code expects: "
            f"{EXPECTED_XGB_FEATURES}"
        )

    print(
        "XGBoost model feature check: PASS"
    )

    # ========================================================
    # BUILD EXACT 206 FEATURES
    # ========================================================

    xgb_features = (
        build_xgboost_features(
            CANDIDATE,
            tfidf,
            svd,
            riasec_scaler
        )
    )

    # ========================================================
    # XGBOOST PROBABILITIES
    # ========================================================

    xgb_scores = (
        compute_xgboost_scores(
            xgb_features,
            xgb_model,
            label_encoder,
            career_order
        )
    )

    # ========================================================
    # FINAL ARRAY ALIGNMENT CHECK
    # ========================================================

    print("\n" + "=" * 75)
    print("ALIGNMENT VERIFICATION")
    print("=" * 75)

    print(
        "Career labels       :",
        len(career_order)
    )

    print(
        "SBERT scores        :",
        len(semantic_scores)
    )

    print(
        "Skill scores        :",
        len(skill_scores)
    )

    print(
        "XGBoost scores      :",
        len(xgb_scores)
    )

    if not (
        len(career_order)
        ==
        len(semantic_scores)
        ==
        len(skill_scores)
        ==
        len(xgb_scores)
    ):

        raise ValueError(
            "FINAL CAREER ALIGNMENT FAILED."
        )

    print(
        "Career / SBERT / Skill / XGBoost "
        "alignment: PASS"
    )

    # ========================================================
    # HYBRID RANKING
    # ========================================================

    (
        final_scores,
        semantic_normalized,
        skill_normalized,
        xgb_normalized
    ) = calculate_final_scores(
        semantic_scores,
        skill_scores,
        xgb_scores
    )

    # ========================================================
    # CONFIDENCE SCORES
    # ========================================================

    confidence_scores = (
        calculate_confidence_scores(
            final_scores
        )
    )

    # ========================================================
    # RANK CAREERS
    # ========================================================

    ranking_indices = np.argsort(
        final_scores
    )[::-1][:TOP_K]

    results = []

    for rank, index in enumerate(
        ranking_indices,
        start=1
    ):

        career = career_order[
            index
        ]

        matched_skills = (
            matched_skills_by_career[
                career
            ]
        )

        skill_alignment = (
            calculate_skill_alignment(
                candidate_skills,
                matched_skills
            )
        )

        result = {

            "rank":
                rank,

            "career":
                career,

            "final_score":
                round(
                    float(
                        final_scores[index]
                    ),
                    6
                ),

            "confidence_score":
                round(
                    float(
                        confidence_scores[index]
                    ),
                    6
                ),

            "confidence_percentage":
                round(
                    float(
                        confidence_scores[index]
                    ) * 100,
                    2
                ),

            "semantic_score":
                round(
                    float(
                        semantic_scores[index]
                    ),
                    6
                ),

            "semantic_normalized":
                round(
                    float(
                        semantic_normalized[index]
                    ),
                    6
                ),

            "skill_match":
                round(
                    float(
                        skill_scores[index]
                    ),
                    6
                ),

            "skill_match_normalized":
                round(
                    float(
                        skill_normalized[index]
                    ),
                    6
                ),

            "xgboost_probability":
                round(
                    float(
                        xgb_scores[index]
                    ),
                    8
                ),

            "xgboost_normalized":
                round(
                    float(
                        xgb_normalized[index]
                    ),
                    6
                ),

            "matched_skills":
                matched_skills,

            "skill_alignment":
                skill_alignment
        }

        results.append(
            result
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    display_results(
        results
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    print("\n" + "=" * 75)
    print("12. SANITY CHECK")
    print("=" * 75)

    duplicate_found = False

    for result in results:

        skills = result[
            "matched_skills"
        ]

        if len(skills) != len(
            set(skills)
        ):

            duplicate_found = True

            print(
                "DUPLICATE FOUND:",
                result["career"]
            )

    if not duplicate_found:

        print(
            "Duplicate matched skills: NONE"
        )

    # --------------------------------------------------------
    # Check final score range
    # --------------------------------------------------------

    if (
        np.min(final_scores) < 0
        or
        np.max(final_scores) > 1
    ):

        raise ValueError(
            "Final scores are outside "
            "expected 0-1 range."
        )

    print(
        "Final score range check: PASS"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_directory = save_results(
        results,
        CANDIDATE
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 75)
    print("V3 RECOMMENDATION COMPLETE")
    print("=" * 75)

    print(
        "\nExisting models were NOT modified."
    )

    print(
        "Existing dataset was NOT modified."
    )

    print(
        "Existing SBERT embeddings were NOT modified."
    )

    print(
        "Existing recommendation files were "
        "NOT overwritten."
    )

    print(
        "\nNew results:",
        output_directory
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
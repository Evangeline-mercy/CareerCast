# ============================================================
# CAREERCAST - MILESTONE 2
# V4 - IMPROVED HYBRID CAREER RECOMMENDER
#
# READ-ONLY INFERENCE MODE
#
# Uses:
#   1. SBERT semantic similarity
#   2. IDF-weighted skill matching
#   3. Existing XGBoost probability
#
# XGBoost:
#   200 SVD features
#   + 6 RIASEC features
#   = 206 features
#
# IMPORTANT:
#   - Dataset is NOT modified
#   - XGBoost is NOT retrained
#   - SBERT is NOT modified
#   - Existing embeddings are NOT modified
#   - Existing V3 results are NOT overwritten
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

DATASET_PATH = "results/careercast_milestone2_dataset.csv"

TFIDF_PATH = "results/tree_features/tfidf_vectorizer.joblib"
SVD_PATH = "results/tree_features/svd.joblib"
RIASEC_SCALER_PATH = "results/tree_features/riasec_scaler.joblib"

XGBOOST_PATH = "results/xgboost_model/xgboost_model.joblib"
XGBOOST_LABEL_ENCODER_PATH = (
    "results/xgboost_model/label_encoder.joblib"
)

SBERT_MODEL_NAME = (
    "results/semantic_embeddings/sbert_finetuned"
)

CAREER_EMBEDDINGS_PATH = (
    "results/semantic_embeddings/"
    "finetuned_career_embeddings.npy"
)

CAREER_LABELS_PATH = (
    "results/semantic_embeddings/"
    "finetuned_career_labels.csv"
)

OUTPUT_ROOT = "v4_output"


# ============================================================
# HYBRID WEIGHTS
# ============================================================

WEIGHT_SEMANTIC = 0.50
WEIGHT_SKILL = 0.40
WEIGHT_XGBOOST = 0.10

TOP_K = 15


# ============================================================
# EXPECTED MODEL CONFIGURATION
# ============================================================

EXPECTED_CAREERS = 878
EXPECTED_XGBOOST_FEATURES = 206
EXPECTED_SVD_FEATURES = 200
EXPECTED_RIASEC_FEATURES = 6


# ============================================================
# CANDIDATE
#
# This can later be replaced by frontend/API input.
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
        "Conventional": 8
    }
}


# ============================================================
# DATASET SKILL EQUIVALENCES
# ============================================================

DATASET_SKILL_EQUIVALENTS = {

    "sql": {
        "structured query language sql",
        "structure query language sql",
        "sql",
    },

    "python": {
        "python",
    },

    "data analysis": {
        "data analysis",
        "statistical analysis software",
        "statistical analysis",
        "data visualization software",
        "data analysis software",
        "data analytics",
        "data processing software",
        "data management software",
    },

    "machine learning": {
        "machine learning",
        "machine learning software",
        "computer modeling software",
        "statistical analysis software",
        "data analysis software",
        "predictive modeling software",
        "data mining software",
        "artificial intelligence software",
        "pattern recognition software",
    },

    "pandas": {
        "python",
        "data analysis software",
        "data processing software",
        "statistical analysis software",
    },

    "numpy": {
        "python",
        "mathematical software",
        "scientific software",
        "statistical analysis software",
        "data analysis software",
    },

    "tensorflow": {
        "machine learning",
        "machine learning software",
        "artificial intelligence software",
        "computer modeling software",
        "predictive modeling software",
    },
}


# ============================================================
# STEP 0
# NORMALIZATION
# ============================================================

def normalize_skill(skill):

    if not isinstance(skill, str):
        return ""

    skill = skill.lower().strip()

    skill = re.sub(
        r"[^a-z0-9\s]",
        " ",
        skill
    )

    skill = re.sub(
        r"\s+",
        " ",
        skill
    ).strip()

    return skill


# ============================================================
# PARSE SKILL STRING
# ============================================================

def parse_skill_string(raw):

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

    print("\n" + "=" * 75)
    print("1. LOADING DATASET")
    print("=" * 75)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [

        "Career",
        "Essential_Skills",
        "Software_Skills",
        "Education_Level",
        "Experience_Level",
        "Realistic",
        "Investigative",
        "Artistic",
        "Social",
        "Enterprising",
        "Conventional",
        "Job_Description"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Dataset rows      :", len(df))
    print("Unique careers    :", df["Career"].nunique())
    print("Required columns  : PASS")

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

        career = str(row["Career"])

        essential_skills = parse_skill_string(
            row["Essential_Skills"]
        )

        software_skills = parse_skill_string(
            row["Software_Skills"]
        )

        career_skills[career].update(
            essential_skills | software_skills
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
    print("3. BUILDING STABLE SKILL WEIGHTS")
    print("=" * 75)

    number_of_careers = len(career_skill_sets)

    document_frequency = defaultdict(int)

    for skills in career_skill_sets.values():

        for skill in skills:
            document_frequency[skill] += 1

    idf = {}

    for skill, frequency in document_frequency.items():

        idf[skill] = (
            math.log(
                (1 + number_of_careers)
                /
                (1 + frequency)
            )
            + 1.0
        )

    print("Unique skills:", len(idf))

    return idf


# ============================================================
# STEP 4
# SKILL SIMILARITY
# ============================================================

def skill_similarity(candidate_skill, career_skill):

    candidate_skill = normalize_skill(
        candidate_skill
    )

    career_skill = normalize_skill(
        career_skill
    )

    if not candidate_skill or not career_skill:
        return 0.0

    # --------------------------------------------------------
    # EXACT MATCH
    # --------------------------------------------------------

    if candidate_skill == career_skill:
        return 1.0

    # --------------------------------------------------------
    # EXPLICIT EQUIVALENCE
    # --------------------------------------------------------

    equivalents = DATASET_SKILL_EQUIVALENTS.get(
        candidate_skill,
        set()
    )

    if career_skill in equivalents:

        if candidate_skill in {
            "python",
            "sql"
        }:
            return 1.0

        if candidate_skill in {
            "machine learning",
            "data analysis"
        }:
            return 0.60

        if candidate_skill in {
            "pandas",
            "numpy",
            "tensorflow"
        }:
            return 0.50

    # --------------------------------------------------------
    # SQL FAMILY
    # --------------------------------------------------------

    sql_family = {

        "sql",
        "structured query language",
        "structured query language sql",
        "database software",
        "database management software",
        "relational database",
        "oracle database",
        "mysql",
        "postgresql",
    }

    if candidate_skill == "sql":
        if career_skill in sql_family:
            return 0.90

    # --------------------------------------------------------
    # PYTHON FAMILY
    # --------------------------------------------------------

    python_family = {

        "python",
        "python programming",
        "programming software",
        "software development tools",
        "computer programming software",
    }

    if candidate_skill == "python":
        if career_skill in python_family:
            return 0.90

    # --------------------------------------------------------
    # MACHINE LEARNING / AI FAMILY
    # --------------------------------------------------------

    ml_candidate_terms = {

        "machine learning",
        "deep learning",
        "artificial intelligence",
        "ai",
        "predictive modeling",
        "machine learning models",
    }

    ml_career_terms = {

        "machine learning software",
        "machine learning models",
        "predictive analytics software",
        "predictive modeling software",
        "data mining software",
        "statistical analysis software",
        "statistical software",
        "computer modeling software",
        "modeling software",
        "data analysis software",
        "data visualization software",
    }

    if candidate_skill in ml_candidate_terms:

        if career_skill in ml_career_terms:
            return 0.65

    # --------------------------------------------------------
    # DATA ANALYSIS FAMILY
    # --------------------------------------------------------

    data_candidate_terms = {

        "data analysis",
        "data analytics",
        "data preprocessing",
        "data visualization",
        "statistical analysis",
    }

    data_career_terms = {

        "data analysis software",
        "data visualization software",
        "statistical analysis software",
        "statistical software",
        "data mining software",
        "business intelligence software",
        "database software",
    }

    if candidate_skill in data_candidate_terms:

        if career_skill in data_career_terms:
            return 0.60

    # --------------------------------------------------------
    # PYTHON ECOSYSTEM
    # --------------------------------------------------------

    if candidate_skill in {
        "pandas",
        "numpy",
        "tensorflow",
    }:

        supporting_terms = {

            "data analysis software",
            "data visualization software",
            "statistical software",
            "statistical analysis software",
            "data mining software",
            "machine learning software",
            "predictive analytics software",
            "computer modeling software",
            "modeling software",
            "software development tools",
        }

        if career_skill in supporting_terms:
            return 0.45

    # --------------------------------------------------------
    # CLOUD / DATA ENGINEERING FAMILY
    # --------------------------------------------------------

    cloud_data_terms = {

        "amazon web services aws software",
        "amazon web services aws sagemaker",
        "amazon redshift",
        "amazon dynamodb",
        "amazon simple storage service s3",
        "apache hadoop",
        "apache hive",
        "apache kafka",
        "apache cassandra",
        "apache spark",
        "data engineering software",
        "cloud computing software",
    }

    if candidate_skill in {

        "machine learning",
        "data analysis",
        "python",
        "pandas",
        "numpy",
        "tensorflow",

    }:

        if career_skill in cloud_data_terms:
            return 0.40

    # --------------------------------------------------------
    # TOKEN OVERLAP
    # --------------------------------------------------------

    candidate_tokens = set(
        candidate_skill.split()
    )

    career_tokens = set(
        career_skill.split()
    )

    if not candidate_tokens or not career_tokens:
        return 0.0

    intersection = (
        candidate_tokens & career_tokens
    )

    if not intersection:
        return 0.0

    candidate_coverage = (
        len(intersection)
        /
        len(candidate_tokens)
    )

    career_coverage = (
        len(intersection)
        /
        len(career_tokens)
    )

    if candidate_coverage < 0.50:
        return 0.0

    score = (
        0.70 * candidate_coverage
        +
        0.30 * career_coverage
    )

    return min(1.0, score)


# ============================================================
# STEP 5
# CALCULATE SKILL MATCH
# ============================================================

def calculate_skill_match(
    candidate_skills,
    career_skills,
    idf
):

    if not candidate_skills or not career_skills:
        return 0.0, []

    matched_details = []

    for candidate_skill in candidate_skills:

        best_score = 0.0
        best_skill = None

        for career_skill in career_skills:

            similarity = skill_similarity(
                candidate_skill,
                career_skill
            )

            if similarity > best_score:

                best_score = similarity
                best_skill = career_skill

        if (
            best_skill is not None
            and best_score >= 0.60
        ):

            matched_details.append(
                (
                    candidate_skill,
                    best_skill,
                    best_score
                )
            )

    if not matched_details:
        return 0.0, []

    candidate_weights = {}

    for skill in candidate_skills:

        candidate_weights[skill] = idf.get(
            skill,
            1.0
        )

    total_weight = sum(
        candidate_weights.values()
    )

    if total_weight <= 0:
        return 0.0, []

    matched_weight = 0.0

    for (
        candidate_skill,
        career_skill,
        match_strength
    ) in matched_details:

        matched_weight += (
            candidate_weights[candidate_skill]
            *
            match_strength
        )

    weighted_coverage = (
        matched_weight
        /
        total_weight
    )

    simple_coverage = (
        len(matched_details)
        /
        len(candidate_skills)
    )

    score = (
        0.70 * weighted_coverage
        +
        0.30 * simple_coverage
    )

    score = max(
        0.0,
        min(1.0, score)
    )

    matched_skills = sorted(
        [
            item[0]
            for item in matched_details
        ]
    )

    return score, matched_skills


# ============================================================
# STEP 6
# LOAD SBERT
# ============================================================

def load_sbert():

    print("\n" + "=" * 75)
    print("4. LOADING SENTENCE-BERT")
    print("=" * 75)

    from sentence_transformers import SentenceTransformer

    print("Model:", SBERT_MODEL_NAME)

    model = SentenceTransformer(
        SBERT_MODEL_NAME
    )

    print(
        "Sentence-BERT loaded successfully."
    )

    return model


# ============================================================
# STEP 7
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
            "Career embeddings not found: "
            f"{CAREER_EMBEDDINGS_PATH}"
        )

    if not os.path.exists(
        CAREER_LABELS_PATH
    ):
        raise FileNotFoundError(
            "Career labels not found: "
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
                "Could not identify career label column."
            )

    career_order = (
        labels_df[label_column]
        .astype(str)
        .tolist()
    )

    if len(career_order) != embeddings.shape[0]:

        raise ValueError(
            "Career labels and embedding rows "
            "do not match."
        )

    if len(career_order) != EXPECTED_CAREERS:

        raise ValueError(
            f"Expected {EXPECTED_CAREERS} careers, "
            f"got {len(career_order)}"
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
# STEP 8
# VERIFY CAREER ALIGNMENT
# ============================================================

def verify_career_alignment(
    career_skill_sets,
    career_order,
    career_embeddings
):

    if len(career_order) != (
        career_embeddings.shape[0]
    ):

        raise ValueError(
            "Career order and embedding rows "
            "are misaligned."
        )

    missing = [

        career

        for career in career_order

        if career not in career_skill_sets
    ]

    if missing:

        raise ValueError(
            "Careers missing from dataset: "
            f"{missing[:10]}"
        )

    print(
        "Dataset / SBERT career alignment: PASS"
    )


# ============================================================
# STEP 9
# COMPUTE SBERT SEMANTIC SCORES
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

    candidate_embedding = sbert_model.encode(
        [candidate_text],
        normalize_embeddings=True
    )

    career_norm = (
        career_embeddings
        /
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

    if len(similarities) != len(
        career_order
    ):

        raise ValueError(
            "SBERT similarity count does "
            "not match career count."
        )

    print(
        "Similarity count:",
        len(similarities)
    )

    print(
        "Minimum:",
        round(float(similarities.min()), 6)
    )

    print(
        "Maximum:",
        round(float(similarities.max()), 6)
    )

    print(
        "SBERT career alignment: PASS"
    )

    return similarities


# ============================================================
# STEP 10
# BUILD EXACT 206 XGBOOST FEATURES
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

    if svd_vector.shape[1] != EXPECTED_SVD_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_SVD_FEATURES} "
            f"SVD features, got "
            f"{svd_vector.shape[1]}"
        )

    # --------------------------------------------------------
    # RIASEC
    # --------------------------------------------------------

    riasec_order = [

        "Realistic",
        "Investigative",
        "Artistic",
        "Social",
        "Enterprising",
        "Conventional"
    ]

    riasec_values = np.array(
        [[
            candidate["riasec"][key]
            for key in riasec_order
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

    if riasec_scaled.shape[1] != EXPECTED_RIASEC_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_RIASEC_FEATURES} "
            f"RIASEC features, got "
            f"{riasec_scaled.shape[1]}"
        )

    # --------------------------------------------------------
    # Final 206 features
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

    if features.shape[1] != EXPECTED_XGBOOST_FEATURES:

        raise ValueError(
            "XGBoost feature mismatch! "
            f"Expected {EXPECTED_XGBOOST_FEATURES}, "
            f"got {features.shape[1]}"
        )

    print(
        "206-feature verification: PASS"
    )

    return features


# ============================================================
# STEP 11
# XGBOOST INFERENCE
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
        )[0]
    )

    print(
        "Probability count:",
        len(probabilities)
    )

    if len(probabilities) != EXPECTED_CAREERS:

        raise ValueError(
            f"Expected {EXPECTED_CAREERS} "
            f"XGBoost probabilities, "
            f"got {len(probabilities)}"
        )

    class_names = (
        label_encoder.inverse_transform(
            np.arange(
                len(probabilities)
            )
        )
    )

    probability_by_career = dict(
        zip(
            class_names,
            probabilities
        )
    )

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
# STEP 12
# SCORE ALIGNMENT VERIFICATION
# ============================================================

def verify_score_alignment(
    career_order,
    semantic_scores,
    skill_scores,
    xgb_scores
):

    print("\n" + "=" * 75)
    print("ALIGNMENT VERIFICATION")
    print("=" * 75)

    expected = len(career_order)

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

    if len(semantic_scores) != expected:
        raise ValueError(
            "SBERT score alignment failed."
        )

    if len(skill_scores) != expected:
        raise ValueError(
            "Skill score alignment failed."
        )

    if len(xgb_scores) != expected:
        raise ValueError(
            "XGBoost score alignment failed."
        )

    print(
        "Career / SBERT / Skill / "
        "XGBoost alignment: PASS"
    )


# ============================================================
# STEP 13
# ROBUST NORMALIZATION
# ============================================================

def robust_normalize(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) == 0:
        return values

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum - minimum < 1e-12:
        return np.zeros_like(values)

    normalized = (
        values - minimum
    ) / (
        maximum - minimum
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    )


# ============================================================
# STEP 14
# HYBRID SCORING
# ============================================================

def calculate_final_scores(
    semantic_scores,
    skill_scores,
    xgb_scores
):

    print("\n" + "=" * 75)
    print("9. HYBRID SCORING")
    print("=" * 75)

    semantic_normalized = robust_normalize(
        semantic_scores
    )

    skill_normalized = robust_normalize(
        skill_scores
    )

    xgb_normalized = robust_normalize(
        xgb_scores
    )

    final_scores = (

        WEIGHT_SEMANTIC
        *
        semantic_normalized

        +

        WEIGHT_SKILL
        *
        skill_normalized

        +

        WEIGHT_XGBOOST
        *
        xgb_normalized
    )

    weight_sum = (
        WEIGHT_SEMANTIC
        +
        WEIGHT_SKILL
        +
        WEIGHT_XGBOOST
    )

    print(
        "Semantic weight :",
        WEIGHT_SEMANTIC
    )

    print(
        "Skill weight    :",
        WEIGHT_SKILL
    )

    print(
        "XGBoost weight  :",
        WEIGHT_XGBOOST
    )

    print(
        "Weight sum      :",
        weight_sum
    )

    if not math.isclose(
        weight_sum,
        1.0,
        rel_tol=1e-9
    ):

        raise ValueError(
            "Hybrid weights must sum to 1."
        )

    return (
        final_scores,
        semantic_normalized,
        skill_normalized,
        xgb_normalized
    )


# ============================================================
# STEP 15
# RECOMMENDATION CONFIDENCE
# ============================================================

def calculate_recommendation_confidence(
    final_scores
):

    scores = np.asarray(
        final_scores,
        dtype=np.float64
    )

    if len(scores) == 0:
        return np.array([])

    # --------------------------------------------------------
    # Percentile-based relative confidence
    #
    # This is NOT a probability.
    # It represents the relative strength of a recommendation
    # compared with the other career scores.
    # --------------------------------------------------------

    low = np.percentile(
        scores,
        5
    )

    high = np.percentile(
        scores,
        95
    )

    if high - low < 1e-12:

        minimum = np.min(scores)
        maximum = np.max(scores)

        if maximum - minimum < 1e-12:
            confidence = np.zeros_like(scores)

        else:

            confidence = (
                scores - minimum
            ) / (
                maximum - minimum
            )

    else:

        confidence = (

            np.clip(
                scores,
                low,
                high
            )
            -
            low
        ) / (
            high - low
        )

    confidence = (
        confidence * 100.0
    )

    return np.clip(
        confidence,
        0.0,
        100.0
    )


# ============================================================
# STEP 16
# SKILL ALIGNMENT
# ============================================================

def calculate_skill_alignment(
    candidate_skills,
    matched_skills
):

    if not candidate_skills:
        return 0.0

    return (
        len(matched_skills)
        /
        len(candidate_skills)
    ) * 100.0


# ============================================================
# STEP 17
# DISPLAY RESULTS
# ============================================================

def display_results(results):

    print("\n" + "=" * 75)

    print(
        f"TOP {len(results)} CAREER RECOMMENDATIONS"
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
            "   RECOMMENDATION SCORE : "
            f"{result['final_score']:.4f}"
        )

        print(
            "   RELATIVE CONFIDENCE  : "
            f"{result['confidence_percentage']:.2f}%"
        )

        print(
            "   Semantic Score       : "
            f"{result['semantic_score']:.4f}"
        )

        print(
            "   Skill Match          : "
            f"{result['skill_match']:.4f}"
        )

        print(
            "   Skill Alignment      : "
            f"{result['skill_alignment']:.2f}%"
        )

        print(
            "   XGBoost Probability  : "
            f"{result['xgboost_probability']:.6f}"
        )

        print(
            "   Matched Skills       : "
            f"{matched}"
        )

    print("\n" + "=" * 75)


# ============================================================
# STEP 18
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    candidate
):

    print("\n" + "=" * 75)
    print("10. SAVING RESULTS")
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
        output_directory,
        exist_ok=True
    )

    json_path = os.path.join(
        output_directory,
        "v4_recommendations.json"
    )

    csv_path = os.path.join(
        output_directory,
        "v4_recommendations.csv"
    )

    output_data = {

        "version": "V4",

        "weights": {

            "semantic":
                WEIGHT_SEMANTIC,

            "skill":
                WEIGHT_SKILL,

            "xgboost":
                WEIGHT_XGBOOST
        },

        "xgboost_features": {

            "svd_features":
                EXPECTED_SVD_FEATURES,

            "riasec_features":
                EXPECTED_RIASEC_FEATURES,

            "total_features":
                EXPECTED_XGBOOST_FEATURES
        },

        "candidate":
            candidate,

        "results":
            results
    }

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
# STEP 19
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("CAREERCAST MILESTONE 2 - V4")
    print("HYBRID CAREER RECOMMENDER")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Build career skill sets
    # --------------------------------------------------------

    career_skill_sets = (
        build_career_skill_sets(df)
    )

    # --------------------------------------------------------
    # 3. Build IDF
    # --------------------------------------------------------

    idf = build_skill_idf(
        career_skill_sets
    )

    # --------------------------------------------------------
    # 4. Candidate skills
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. Load career embeddings
    # --------------------------------------------------------

    (
        career_embeddings,
        career_order
    ) = load_career_embeddings()

    # --------------------------------------------------------
    # 6. Verify career alignment
    # --------------------------------------------------------

    verify_career_alignment(

        career_skill_sets,

        career_order,

        career_embeddings
    )

    # --------------------------------------------------------
    # 7. Calculate skill scores
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 8. Load SBERT
    # --------------------------------------------------------

    sbert_model = load_sbert()

    # --------------------------------------------------------
    # 9. Compute semantic scores
    # --------------------------------------------------------

    semantic_scores = (
        compute_semantic_scores(

            CANDIDATE,

            career_embeddings,

            career_order,

            sbert_model
        )
    )

    # --------------------------------------------------------
    # 10. Load XGBoost artifacts
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("7. LOADING XGBOOST ARTIFACTS")
    print("=" * 75)

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
        len(tfidf.vocabulary_)
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
        len(label_encoder.classes_)
    )

    # --------------------------------------------------------
    # 11. Verify XGBoost
    # --------------------------------------------------------

    if (
        xgb_model.n_features_in_
        != EXPECTED_XGBOOST_FEATURES
    ):

        raise ValueError(
            "Existing XGBoost model does not "
            f"expect {EXPECTED_XGBOOST_FEATURES} features."
        )

    if (
        len(label_encoder.classes_)
        != EXPECTED_CAREERS
    ):

        raise ValueError(
            f"Expected {EXPECTED_CAREERS} "
            "XGBoost classes."
        )

    print(
        "XGBoost model feature check: PASS"
    )

    # --------------------------------------------------------
    # 12. Build 206 features
    # --------------------------------------------------------

    xgb_features = (
        build_xgboost_features(

            CANDIDATE,

            tfidf,

            svd,

            riasec_scaler
        )
    )

    # --------------------------------------------------------
    # 13. XGBoost probabilities
    # --------------------------------------------------------

    xgb_scores = (
        compute_xgboost_scores(

            xgb_features,

            xgb_model,

            label_encoder,

            career_order
        )
    )

    # --------------------------------------------------------
    # 14. Alignment
    # --------------------------------------------------------

    verify_score_alignment(

        career_order,

        semantic_scores,

        skill_scores,

        xgb_scores
    )

    # --------------------------------------------------------
    # 15. Hybrid ranking
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 16. Confidence
    # --------------------------------------------------------

    confidence_scores = (
        calculate_recommendation_confidence(

            final_scores
        )
    )

    # --------------------------------------------------------
    # 17. Rank careers
    # --------------------------------------------------------

    ranking_indices = (
        np.argsort(
            final_scores
        )[::-1][:TOP_K]
    )

    # --------------------------------------------------------
    # 18. BUILD REAL RESULTS
    #
    # IMPORTANT:
    # Do NOT replace model career names with fake/demo names.
    # --------------------------------------------------------

    results = []

    for rank, index in enumerate(
        ranking_indices,
        start=1
    ):

        career = career_order[index]

        matched = (
            matched_skills_by_career.get(
                career,
                []
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

            "confidence_percentage":
                round(
                    float(
                        confidence_scores[index]
                    ),
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
                matched,

            "skill_alignment":
                round(
                    calculate_skill_alignment(
                        candidate_skills,
                        matched
                    ),
                    2
                )
        }

        results.append(
            result
        )

    # --------------------------------------------------------
    # 19. Display
    # --------------------------------------------------------

    display_results(
        results
    )

    # --------------------------------------------------------
    # 20. Sanity checks
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("11. SANITY CHECK")
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
    # Score range
    # --------------------------------------------------------

    if (

        np.all(
            final_scores >= 0.0
        )

        and

        np.all(
            final_scores <= 1.0
        )
    ):

        print(
            "Final score range check: PASS"
        )

    else:

        raise ValueError(
            "Final score range check FAILED."
        )

    # --------------------------------------------------------
    # Result count
    # --------------------------------------------------------

    if len(results) != TOP_K:

        raise ValueError(
            f"Expected {TOP_K} recommendations, "
            f"got {len(results)}"
        )

    print(
        f"Top-{TOP_K} result count check: PASS"
    )

    # --------------------------------------------------------
    # 21. Save
    # --------------------------------------------------------

    output_directory = (
        save_results(
            results,
            CANDIDATE
        )
    )

    # --------------------------------------------------------
    # 22. Final status
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("V4 RECOMMENDATION COMPLETE")
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
        "Existing V3 recommendation files "
        "were NOT overwritten."
    )

    print(
        "\nNew results:",
        output_directory
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
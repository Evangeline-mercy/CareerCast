# -*- coding: utf-8 -*-

"""
CareerCast Milestone 2
READ-ONLY XGBOOST PREDICTION INVESTIGATION

Purpose:
    Investigate why a candidate with Python / ML / TensorFlow skills
    may receive an unexpected XGBoost career prediction.

IMPORTANT:
    - Does NOT retrain the model.
    - Does NOT modify any model.
    - Does NOT overwrite existing files.
    - Does NOT modify career_recommender.py.
    - Does NOT modify build_hybrid_recommender.py.
    - Only reads existing artifacts and prints diagnostics.

Run:
    python investigate_xgb_prediction.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "results/careercast_candidate_profiles.csv"

TFIDF_PATH = "results/tree_features/tfidf_vectorizer.joblib"
SVD_PATH = "results/tree_features/svd.joblib"
RIASEC_SCALER_PATH = "results/tree_features/riasec_scaler.joblib"
LABEL_ENCODER_PATH = "results/tree_features/label_encoder.joblib"

XGB_MODEL_PATH = "results/xgboost_model/xgboost_model.joblib"

SEMANTIC_METADATA = (
    "results/semantic_embeddings/profile_metadata.csv"
)


# ============================================================
# CANDIDATE
# ============================================================

CANDIDATE = {
    "Skills": (
        "Python, SQL, Machine Learning, Pandas, NumPy, "
        "Data Analysis, TensorFlow"
    ),

    "Education": (
        "B.E. Electronics and Communication Engineering"
    ),

    "Experience": (
        "6 months internship experience in Python, data analysis "
        "and machine learning projects"
    ),

    "Job_Description": (
        "Develop machine learning models, analyze datasets, "
        "build predictive systems, perform data preprocessing "
        "and visualization, and deploy data-driven applications."
    ),

    "Realistic": 6,
    "Investigative": 9,
    "Artistic": 4,
    "Social": 5,
    "Enterprising": 5,
    "Conventional": 8,
}


# ============================================================
# HELPERS
# ============================================================

def separator():
    print("=" * 75)


def section(title):
    print()
    separator()
    print(title)
    separator()


def check_file(path):
    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {path}")
        return False

    print(f"[FOUND] {path}")
    return True


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# START
# ============================================================

section("CAREERCAST MILESTONE 2")
print("READ-ONLY XGBOOST PREDICTION INVESTIGATION")
print()
print("No model training will be performed.")
print("No existing artifact will be modified.")


# ============================================================
# 1. CHECK ARTIFACTS
# ============================================================

section("1. CHECKING REQUIRED ARTIFACTS")

required_files = [
    DATASET,
    TFIDF_PATH,
    SVD_PATH,
    RIASEC_SCALER_PATH,
    LABEL_ENCODER_PATH,
    XGB_MODEL_PATH,
]

all_present = True

for path in required_files:
    if not check_file(path):
        all_present = False

if not all_present:
    print()
    print("[STOP] One or more required artifacts are missing.")
    sys.exit(1)


# ============================================================
# 2. LOAD ARTIFACTS
# ============================================================

section("2. LOADING EXISTING ARTIFACTS")

try:
    df = pd.read_csv(DATASET)

    tfidf = joblib.load(TFIDF_PATH)
    svd = joblib.load(SVD_PATH)
    riasec_scaler = joblib.load(RIASEC_SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)

except Exception as e:
    print("[ERROR] Failed to load artifacts.")
    print(e)
    sys.exit(1)

print(f"Dataset rows          : {len(df)}")
print(f"Dataset columns       : {list(df.columns)}")

print(f"TF-IDF vocabulary     : {len(tfidf.vocabulary_)}")
print(f"SVD components        : {svd.n_components}")

print(
    f"Label encoder classes : "
    f"{len(label_encoder.classes_)}"
)

print(
    f"XGBoost classes       : "
    f"{len(xgb_model.classes_)}"
)


# ============================================================
# 3. DATASET STRUCTURE
# ============================================================

section("3. DATASET STRUCTURE")

required_columns = [
    "Career",
    "Skills",
    "Education",
    "Experience",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("[ERROR] Missing required dataset columns:")
    print(missing_columns)
    sys.exit(1)

print("Required columns found.")

print()
print("Number of unique careers:")
print(df["Career"].nunique())

print()
print("Top career frequencies:")

career_counts = (
    df["Career"]
    .value_counts()
    .head(20)
)

print(career_counts.to_string())


# ============================================================
# 4. BUILD EXACT TEXT INPUT
# ============================================================

section("4. BUILDING XGBOOST TEXT FEATURES")

text = (
    CANDIDATE["Skills"] + " "
    + CANDIDATE["Education"] + " "
    + CANDIDATE["Experience"]
)

print("Text supplied to TF-IDF/SVD:")
print()
print(text)


# ============================================================
# 5. CHECK VOCABULARY COVERAGE
# ============================================================

section("5. CHECKING TF-IDF VOCABULARY COVERAGE")

candidate_tokens = [
    "python",
    "sql",
    "machine",
    "learning",
    "pandas",
    "numpy",
    "data",
    "analysis",
    "tensorflow",
    "electronics",
    "communication",
    "engineering",
]

vocabulary = tfidf.vocabulary_

found_tokens = []
missing_tokens = []

for token in candidate_tokens:

    if token.lower() in vocabulary:
        found_tokens.append(token)
    else:
        missing_tokens.append(token)

print("Expected important tokens:")
print(candidate_tokens)

print()
print("Tokens FOUND in trained TF-IDF vocabulary:")
print(found_tokens)

print()
print("Tokens NOT FOUND in trained TF-IDF vocabulary:")
print(missing_tokens)

print()
print(
    f"Vocabulary coverage: "
    f"{len(found_tokens)}/{len(candidate_tokens)}"
)


# ============================================================
# 6. TF-IDF REPRESENTATION
# ============================================================

section("6. TF-IDF REPRESENTATION")

try:
    tfidf_matrix = tfidf.transform([text])

except Exception as e:
    print("[ERROR] TF-IDF transformation failed.")
    print(e)
    sys.exit(1)

print("TF-IDF shape:", tfidf_matrix.shape)

nonzero = tfidf_matrix.nnz

print("Non-zero TF-IDF features:", nonzero)

if nonzero == 0:
    print()
    print(
        "[CRITICAL WARNING] The candidate text produced ZERO "
        "TF-IDF features."
    )
    print(
        "This would explain an abnormal XGBoost prediction."
    )


# ============================================================
# 7. SHOW STRONGEST TF-IDF FEATURES
# ============================================================

section("7. STRONGEST TF-IDF FEATURES")

try:

    feature_names = np.array(
        tfidf.get_feature_names_out()
    )

    row = tfidf_matrix.toarray()[0]

    top_indices = np.argsort(row)[::-1]

    shown = 0

    print(
        f"{'Feature':<35}"
        f"{'TF-IDF':>12}"
    )

    print("-" * 50)

    for idx in top_indices:

        if row[idx] <= 0:
            break

        print(
            f"{feature_names[idx]:<35}"
            f"{row[idx]:>12.6f}"
        )

        shown += 1

        if shown >= 30:
            break

    if shown == 0:
        print("No active TF-IDF features.")

except Exception as e:
    print("[WARNING] Could not display TF-IDF features.")
    print(e)


# ============================================================
# 8. SVD
# ============================================================

section("8. SVD TRANSFORMATION")

try:

    svd_matrix = svd.transform(tfidf_matrix)

except Exception as e:
    print("[ERROR] SVD transformation failed.")
    print(e)
    sys.exit(1)

print("SVD shape:", svd_matrix.shape)

print(
    "SVD non-zero values:",
    np.count_nonzero(svd_matrix)
)

print(
    "SVD min:",
    float(np.min(svd_matrix))
)

print(
    "SVD max:",
    float(np.max(svd_matrix))
)


# ============================================================
# 9. RIASEC FEATURES
# ============================================================

section("9. RIASEC FEATURE INSPECTION")

riasec_names = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]

riasec_values = np.array([
    CANDIDATE["Realistic"],
    CANDIDATE["Investigative"],
    CANDIDATE["Artistic"],
    CANDIDATE["Social"],
    CANDIDATE["Enterprising"],
    CANDIDATE["Conventional"],
], dtype=float).reshape(1, -1)

print("Raw RIASEC values:")

for name, value in zip(
    riasec_names,
    riasec_values[0]
):
    print(
        f"  {name:<15}: {value:.2f}"
    )


# ============================================================
# 10. RIASEC SCALING
# ============================================================

section("10. RIASEC SCALING")

try:

    riasec_scaled = riasec_scaler.transform(
        riasec_values
    )

except Exception as e:
    print("[ERROR] RIASEC scaling failed.")
    print(e)
    sys.exit(1)

print("Scaled RIASEC values:")

for name, raw, scaled in zip(
    riasec_names,
    riasec_values[0],
    riasec_scaled[0],
):

    print(
        f"  {name:<15}"
        f" raw={raw:>7.2f}"
        f" scaled={scaled:>12.6f}"
    )


# ============================================================
# 11. FINAL XGBOOST INPUT
# ============================================================

section("11. FINAL XGBOOST FEATURE VECTOR")

X_candidate = np.hstack([
    svd_matrix,
    riasec_scaled,
])

print(
    "Final feature shape:",
    X_candidate.shape
)

expected_features = 206

if X_candidate.shape[1] != expected_features:

    print()
    print(
        f"[WARNING] Expected {expected_features} features "
        f"but generated {X_candidate.shape[1]}."
    )

else:

    print(
        f"Feature count check: PASS ({expected_features})"
    )


# ============================================================
# 12. XGBOOST PREDICTION
# ============================================================

section("12. XGBOOST RAW PREDICTION")

try:

    probabilities = xgb_model.predict_proba(
        X_candidate
    )[0]

except Exception as e:
    print("[ERROR] XGBoost prediction failed.")
    print(e)
    sys.exit(1)

print(
    "Probability vector shape:",
    probabilities.shape
)

print(
    "Probability sum:",
    float(np.sum(probabilities))
)

print(
    "Probability minimum:",
    float(np.min(probabilities))
)

print(
    "Probability maximum:",
    float(np.max(probabilities))
)


# ============================================================
# 13. TOP 20 XGBOOST CAREERS
# ============================================================

section("13. TOP 20 XGBOOST PREDICTIONS")

top_indices = np.argsort(
    probabilities
)[::-1][:20]

print(
    f"{'Rank':<6}"
    f"{'Career':<60}"
    f"{'Probability':>12}"
)

print("-" * 82)

for rank, idx in enumerate(
    top_indices,
    start=1
):

    career = label_encoder.inverse_transform(
        [xgb_model.classes_[idx]]
    )[0]

    probability = probabilities[idx]

    print(
        f"{rank:<6}"
        f"{career:<60}"
        f"{probability:>12.6f}"
    )


# ============================================================
# 14. CHECK SPECIFIC EXPECTED CAREERS
# ============================================================

section("14. CHECKING EXPECTED AI / DATA / SOFTWARE CAREERS")

keywords = [
    "data scientist",
    "machine learning",
    "data analyst",
    "artificial intelligence",
    "software",
    "computer",
    "data",
    "scientist",
    "developer",
    "engineer",
    "programmer",
]

matches = []

for idx, encoded_class in enumerate(
    xgb_model.classes_
):

    career = label_encoder.inverse_transform(
        [encoded_class]
    )[0]

    career_lower = career.lower()

    if any(
        keyword in career_lower
        for keyword in keywords
    ):

        matches.append(
            (
                career,
                float(probabilities[idx])
            )
        )

matches.sort(
    key=lambda x: x[1],
    reverse=True
)

if matches:

    print(
        f"{'Career':<65}"
        f"{'Probability':>12}"
    )

    print("-" * 80)

    for career, probability in matches[:40]:

        print(
            f"{career:<65}"
            f"{probability:>12.6f}"
        )

else:

    print(
        "No matching AI/data/software careers found."
    )


# ============================================================
# 15. FIND DATASET EXAMPLES FOR TOP CAREERS
# ============================================================

section("15. TRAINING DATA EXAMPLES FOR TOP CAREERS")

top_careers = []

for idx in top_indices[:10]:

    encoded_class = xgb_model.classes_[idx]

    career = label_encoder.inverse_transform(
        [encoded_class]
    )[0]

    top_careers.append(career)

for career in top_careers:

    subset = df[
        df["Career"].astype(str).str.strip()
        == str(career).strip()
    ]

    print()
    print("-" * 75)
    print("CAREER:", career)
    print("Dataset rows:", len(subset))
    print("-" * 75)

    if subset.empty:
        print("No matching dataset rows found.")
        continue

    sample_rows = subset.head(3)

    for row_number, (_, row) in enumerate(
        sample_rows.iterrows(),
        start=1
    ):

        print()
        print(f"Example {row_number}:")

        print(
            "Skills:",
            clean_text(row.get("Skills", ""))
        )

        print(
            "Education:",
            clean_text(row.get("Education", ""))
        )

        print(
            "Experience:",
            clean_text(row.get("Experience", ""))
        )

        if "Job_Description" in row:

            print(
                "Job Description:",
                clean_text(
                    row.get("Job_Description", "")
                )[:300]
            )


# ============================================================
# 16. CHECK SPECIFIC CAREER PROBABILITIES
# ============================================================

section("16. SPECIFIC CAREER CHECK")

specific_names = [
    "Data Scientists",
    "Data Scientist",
    "Machine Learning Engineers",
    "Machine Learning Engineer",
    "Data Analysts",
    "Data Analyst",
    "Software Developers",
    "Software Developer",
    "Computer and Information Research Scientists",
    "Computer Network Architects",
    "Sales Engineers",
    "Library Assistants, Clerical",
]

found_specific = False

for target in specific_names:

    matches = [
        (
            idx,
            career,
            probability
        )
        for idx, encoded_class
        in enumerate(xgb_model.classes_)
        for career in [
            label_encoder.inverse_transform(
                [encoded_class]
            )[0]
        ]
        for probability in [
            float(probabilities[idx])
        ]
        if career.lower() == target.lower()
    ]

    if matches:

        found_specific = True

        _, career, probability = matches[0]

        print(
            f"{career:<60}"
            f"{probability:.8f}"
        )

if not found_specific:

    print(
        "None of the exact requested career names "
        "were found."
    )


# ============================================================
# 17. BASIC DIAGNOSIS
# ============================================================

section("17. AUTOMATIC DIAGNOSIS")

max_probability = float(
    np.max(probabilities)
)

top_idx = int(
    np.argmax(probabilities)
)

top_career = label_encoder.inverse_transform(
    [xgb_model.classes_[top_idx]]
)[0]

print(
    "XGBoost Top-1 Career:",
    top_career
)

print(
    "XGBoost Top-1 Probability:",
    f"{max_probability:.6f}"
)

print()

if nonzero == 0:

    print(
        "CRITICAL: Candidate text generated zero "
        "TF-IDF features."
    )

    print(
        "The XGBoost text input is therefore not "
        "receiving meaningful vocabulary information."
    )

elif len(found_tokens) <= 2:

    print(
        "WARNING: Very low TF-IDF vocabulary coverage."
    )

    print(
        "Only a small number of important candidate "
        "tokens were recognized."
    )

else:

    print(
        "TF-IDF vocabulary coverage is not obviously zero."
    )

print()

if max_probability < 0.10:

    print(
        "The XGBoost prediction is highly distributed "
        "across many careers."
    )

    print(
        "Top-1 probability is below 0.10."
    )

else:

    print(
        "XGBoost has a relatively strong top prediction."
    )


# ============================================================
# 18. FINAL STATUS
# ============================================================

section("FINAL STATUS")

print(
    "READ-ONLY INVESTIGATION COMPLETE."
)

print()
print(
    "No model was retrained."
)

print(
    "No model artifact was modified."
)

print(
    "No existing recommendation file was overwritten."
)

print()
print(
    "Use the complete output above to determine whether "
    "the problem originates from:"
)

print(
    "  1. TF-IDF vocabulary coverage"
)

print(
    "  2. SVD representation"
)

print(
    "  3. RIASEC scaling/input"
)

print(
    "  4. XGBoost model behavior"
)

print(
    "  5. Dataset career-label distribution"
)

print(
    "  6. Feature engineering mismatch"
)

separator()
print("DIAGNOSTIC COMPLETE")
separator()
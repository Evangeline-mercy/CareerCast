# -*- coding: utf-8 -*-

"""
CareerCast Milestone 2
Corrected Candidate Inference Diagnostic

IMPORTANT:
- Does NOT retrain XGBoost
- Does NOT modify existing model artifacts
- Does NOT overwrite existing recommendation files
- Uses existing saved artifacts only
- Compares candidate feature representations
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
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

XGB_PATH = os.path.join(
    BASE_DIR,
    "results",
    "xgboost_model",
    "xgboost_model.joblib"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "corrected_candidate_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

section("CAREERCAST MILESTONE 2")
print("CORRECTED CANDIDATE INFERENCE DIAGNOSTIC")
print()
print("READ-ONLY MODE")
print("Existing model files will NOT be modified.")
print("Existing model will NOT be retrained.")
print()

log("Loading existing artifacts...")

dataset = pd.read_csv(DATASET)

tfidf = joblib.load(TFIDF_PATH)
svd = joblib.load(SVD_PATH)
riasec_scaler = joblib.load(RIASEC_SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
xgb_model = joblib.load(XGB_PATH)

log(f"Dataset rows       : {len(dataset)}")
log(f"Dataset columns    : {list(dataset.columns)}")
log(f"TF-IDF vocabulary  : {len(tfidf.vocabulary_)}")
log(f"SVD components     : {svd.n_components}")
log(f"Careers            : {len(label_encoder.classes_)}")


# ============================================================
# CANDIDATE
# ============================================================

candidate_skills = (
    "Python, SQL, Machine Learning, Pandas, NumPy, "
    "Data Analysis, TensorFlow"
)

candidate_education = (
    "B.E. Electronics and Communication Engineering"
)

candidate_experience = (
    "6 months internship experience in Python, data analysis "
    "and machine learning projects"
)

candidate_job_description = (
    "Develop machine learning models, analyze datasets, "
    "build predictive systems, perform data preprocessing "
    "and visualization, and deploy data-driven applications."
)

riasec = np.array([
    6.0,  # Realistic
    9.0,  # Investigative
    4.0,  # Artistic
    5.0,  # Social
    5.0,  # Enterprising
    8.0   # Conventional
], dtype=float)


# ============================================================
# DISPLAY DATASET EDUCATION / EXPERIENCE TYPES
# ============================================================

section("1. TRAINING DATA TYPE INSPECTION")

for col in ["Skills", "Education", "Experience", "Job_Description"]:
    if col in dataset.columns:
        print()
        print(f"{col}:")
        print(f"  dtype       : {dataset[col].dtype}")
        print(f"  sample type : {type(dataset[col].dropna().iloc[0]).__name__}")
        print(f"  sample value: {dataset[col].dropna().iloc[0]}")


# ============================================================
# DATASET EDUCATION / EXPERIENCE STATISTICS
# ============================================================

section("2. TRAINING NUMERIC FEATURE INSPECTION")

for col in ["Education", "Experience"]:
    if col not in dataset.columns:
        continue

    numeric_values = pd.to_numeric(
        dataset[col],
        errors="coerce"
    )

    valid = numeric_values.dropna()

    print()
    print(f"{col}:")
    print(f"  Numeric values : {len(valid)} / {len(dataset)}")

    if len(valid) > 0:
        print(f"  Minimum        : {valid.min():.4f}")
        print(f"  Maximum        : {valid.max():.4f}")
        print(f"  Mean           : {valid.mean():.4f}")
        print(f"  Median         : {valid.median():.4f}")


# ============================================================
# IMPORTANT REPRESENTATION CHECK
# ============================================================

section("3. FEATURE REPRESENTATION CHECK")

education_numeric = pd.to_numeric(
    dataset["Education"],
    errors="coerce"
)

experience_numeric = pd.to_numeric(
    dataset["Experience"],
    errors="coerce"
)

education_numeric_ratio = education_numeric.notna().mean()
experience_numeric_ratio = experience_numeric.notna().mean()

print()
print(
    f"Education numeric ratio : "
    f"{education_numeric_ratio:.4f}"
)

print(
    f"Experience numeric ratio: "
    f"{experience_numeric_ratio:.4f}"
)

if education_numeric_ratio > 0.90:
    print()
    print("WARNING:")
    print(
        "Training Education is predominantly numeric, "
        "but the candidate supplied degree text."
    )

if experience_numeric_ratio > 0.90:
    print()
    print("WARNING:")
    print(
        "Training Experience is predominantly numeric, "
        "but the candidate supplied natural-language experience."
    )


# ============================================================
# FIND TRAINING TEXT CONSTRUCTION
# ============================================================

section("4. RECONSTRUCTING TRAINING TEXT")

sample = dataset.iloc[0]

training_text = (
    str(sample["Skills"]) + " " +
    str(sample["Education"]) + " " +
    str(sample["Experience"])
)

print()
print("Example training text:")
print("-" * 75)
print(training_text[:1000])
print("-" * 75)


# ============================================================
# CANDIDATE TEXT
# ============================================================

candidate_text = (
    candidate_skills + " " +
    candidate_education + " " +
    candidate_experience
)

print()
print("Candidate text:")
print("-" * 75)
print(candidate_text)
print("-" * 75)


# ============================================================
# TF-IDF COVERAGE
# ============================================================

section("5. TF-IDF CANDIDATE COVERAGE")

candidate_lower = candidate_text.lower()

important_terms = [
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
    "internship",
    "projects"
]

found = []
missing = []

for token in important_terms:
    if token in tfidf.vocabulary_:
        found.append(token)
    else:
        missing.append(token)

print()
print("FOUND:")
print(found)

print()
print("NOT FOUND:")
print(missing)

print()
print(
    f"Vocabulary coverage: "
    f"{len(found)}/{len(important_terms)}"
)


# ============================================================
# BUILD XGBOOST FEATURES
# ============================================================

section("6. BUILDING CANDIDATE FEATURES")

tfidf_matrix = tfidf.transform([candidate_text])

print()
print(f"TF-IDF shape     : {tfidf_matrix.shape}")
print(f"TF-IDF non-zero  : {tfidf_matrix.nnz}")

svd_features = svd.transform(tfidf_matrix)

print()
print(f"SVD shape        : {svd_features.shape}")
print(f"SVD min          : {svd_features.min():.8f}")
print(f"SVD max          : {svd_features.max():.8f}")


# ============================================================
# RIASEC
# ============================================================

riasec_scaled = riasec_scaler.transform(
    riasec.reshape(1, -1)
)

print()
print("RIASEC raw:")
for name, value in zip(
    [
        "Realistic",
        "Investigative",
        "Artistic",
        "Social",
        "Enterprising",
        "Conventional"
    ],
    riasec
):
    print(f"  {name:<15}: {value:.4f}")

print()
print("RIASEC scaled:")
for name, value in zip(
    [
        "Realistic",
        "Investigative",
        "Artistic",
        "Social",
        "Enterprising",
        "Conventional"
    ],
    riasec_scaled[0]
):
    print(f"  {name:<15}: {value:.8f}")


# ============================================================
# FINAL VECTOR
# ============================================================

X_candidate = np.hstack([
    svd_features,
    riasec_scaled
])

print()
print(f"Final feature shape: {X_candidate.shape}")

if X_candidate.shape[1] != 206:
    raise RuntimeError(
        f"Expected 206 features, got {X_candidate.shape[1]}"
    )


# ============================================================
# XGBOOST PREDICTION
# ============================================================

section("7. XGBOOST PREDICTION")

probabilities = xgb_model.predict_proba(
    X_candidate
)[0]

print()
print(f"Probability count : {len(probabilities)}")
print(f"Probability sum   : {probabilities.sum():.8f}")
print(f"Maximum probability: {probabilities.max():.8f}")


# ============================================================
# TOP 20
# ============================================================

top_indices = np.argsort(
    probabilities
)[::-1][:20]

rows = []

print()
print(
    f"{'Rank':<6}"
    f"{'Career':<65}"
    f"{'Probability':>12}"
)

print("-" * 85)

for rank, idx in enumerate(top_indices, 1):

    career = label_encoder.inverse_transform(
        [idx]
    )[0]

    probability = float(probabilities[idx])

    print(
        f"{rank:<6}"
        f"{career[:64]:<65}"
        f"{probability:>12.6f}"
    )

    rows.append({
        "Rank": rank,
        "Career": career,
        "Probability": probability
    })


# ============================================================
# TARGET CAREER INSPECTION
# ============================================================

section("8. EXPECTED CAREER INSPECTION")

target_careers = [
    "Software Developers",
    "Web Developers",
    "Computer Systems Analysts",
    "Data Warehousing Specialists",
    "Database Architects",
    "Database Administrators",
    "Computer and Information Research Scientists",
    "Computer Network Architects",
    "Network and Computer Systems Administrators",
    "Telecommunications Engineering Specialists",
    "Sales Engineers"
]

print()
print(
    f"{'Career':<65}"
    f"{'Probability':>12}"
)

print("-" * 80)

target_rows = []

for career in target_careers:

    matches = np.where(
        label_encoder.classes_ == career
    )[0]

    if len(matches) == 0:
        print(f"{career:<65} NOT FOUND")
        continue

    idx = matches[0]
    probability = float(probabilities[idx])

    print(
        f"{career:<65}"
        f"{probability:>12.8f}"
    )

    target_rows.append({
        "Career": career,
        "Probability": probability
    })


# ============================================================
# TRAINING CAREER SIMILARITY CHECK
# ============================================================

section("9. DATASET CAREER MATCH CHECK")

candidate_skill_tokens = set(
    word.strip().lower()
    for word in candidate_skills.replace(",", " ").split()
    if word.strip()
)

print()
print("Candidate skill tokens:")
print(sorted(candidate_skill_tokens))

print()
print(
    "Checking whether important candidate skills "
    "actually occur in expected AI/data careers..."
)

for career in target_careers[:8]:

    career_rows = dataset[
        dataset["Career"].astype(str) == career
    ]

    if len(career_rows) == 0:
        continue

    career_text = " ".join(
        career_rows["Skills"]
        .fillna("")
        .astype(str)
        .tolist()
    ).lower()

    matches = [
        token
        for token in candidate_skill_tokens
        if token in career_text
    ]

    print()
    print(career)
    print(f"  Candidate skill matches: {matches}")


# ============================================================
# SAVE DIAGNOSTIC OUTPUT
# ============================================================

top_df = pd.DataFrame(rows)

top_path = os.path.join(
    OUTPUT_DIR,
    "xgb_top20_corrected_diagnostic.csv"
)

top_df.to_csv(
    top_path,
    index=False,
    encoding="utf-8"
)

summary = {
    "dataset_rows": int(len(dataset)),
    "number_of_careers": int(len(label_encoder.classes_)),
    "tfidf_vocabulary": int(len(tfidf.vocabulary_)),
    "svd_components": int(svd.n_components),
    "final_feature_count": int(X_candidate.shape[1]),
    "education_numeric_ratio": float(education_numeric_ratio),
    "experience_numeric_ratio": float(experience_numeric_ratio),
    "tfidf_found_terms": found,
    "tfidf_missing_terms": missing,
    "top1_career": str(
        label_encoder.inverse_transform(
            [int(top_indices[0])]
        )[0]
    ),
    "top1_probability": float(
        probabilities[top_indices[0]]
    ),
    "candidate_input": {
        "skills": candidate_skills,
        "education": candidate_education,
        "experience": candidate_experience,
        "job_description": candidate_job_description,
        "riasec": riasec.tolist()
    }
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "corrected_candidate_diagnostic.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

section("FINAL DIAGNOSIS")

top1_career = label_encoder.inverse_transform(
    [int(top_indices[0])]
)[0]

top1_probability = float(
    probabilities[top_indices[0]]
)

print()
print(f"Current XGBoost Top-1 : {top1_career}")
print(f"Current probability   : {top1_probability:.8f}")

print()

if education_numeric_ratio > 0.90:
    print(
        "[IMPORTANT] Education representation mismatch detected."
    )

if experience_numeric_ratio > 0.90:
    print(
        "[IMPORTANT] Experience representation mismatch detected."
    )

if top1_career in [
    "Library Assistants, Clerical",
    "Medical Assistants",
    "Pharmacy Aides",
    "File Clerks",
    "Procurement Clerks"
]:
    print(
        "[WARNING] Top prediction is unrelated to the "
        "candidate's stated AI/data/software profile."
    )

print()
print("NO MODEL WAS RETRAINED.")
print("NO MODEL ARTIFACT WAS MODIFIED.")
print("NO EXISTING RECOMMENDATION FILE WAS OVERWRITTEN.")

print()
print("Diagnostic files:")
print(top_path)
print(summary_path)

print()
print("=" * 75)
print("DIAGNOSTIC COMPLETE")
print("=" * 75)
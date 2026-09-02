"""
CareerCast - Milestone 2
Step 1: Prepare compact features for Random Forest and XGBoost

Pipeline:
Skills + Education + Experience
        ↓
TF-IDF
        ↓
TruncatedSVD
        ↓
200 semantic features
        +
6 RIASEC features
        ↓
206-dimensional feature matrix
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


# ==============================================================
# CONFIGURATION
# ==============================================================

DATA_PATH = Path("results/careercast_candidate_profiles.csv")

OUTPUT_DIR = Path("results/tree_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

SVD_COMPONENTS = 200

NUMERIC_COLS = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def main():

    print("=" * 75)
    print("CAREERCAST MILESTONE 2")
    print("STEP 1 - COMPACT FEATURE PREPARATION")
    print("=" * 75)

    # ==========================================================
    # 1. LOAD DATASET
    # ==========================================================

    log("Loading candidate profiles...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {DATA_PATH.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique careers: {df['Career'].nunique()}")

    missing = df.isnull().sum().sum()

    print(f"Total missing values: {missing}")

    if missing > 0:
        raise ValueError(
            "Dataset contains missing values. "
            "Fix the dataset before continuing."
        )

    # ==========================================================
    # 2. CREATE LEAKAGE-RESISTANT TEXT
    # ==========================================================

    log("Creating text features...")

    # IMPORTANT:
    # Job_Description is intentionally excluded.
    #
    # Career and Profile_ID are also excluded.
    #
    # This keeps the experiment consistent with
    # the leakage-resistant baseline.

    df["text_blob"] = (
        df["Skills"].astype(str)
        + " . "
        + df["Education"].astype(str)
        + " . "
        + df["Experience"].astype(str)
    )

    # ==========================================================
    # 3. ENCODE CAREER LABELS
    # ==========================================================

    log("Encoding career labels...")

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["Career"])

    print(f"Number of career classes: {len(label_encoder.classes_)}")

    # ==========================================================
    # 4. STRATIFIED TRAIN/TEST SPLIT
    # ==========================================================

    log("Creating stratified train/test split...")

    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    # ==========================================================
    # 5. TF-IDF
    # ==========================================================

    log("Creating TF-IDF features...")

    tfidf = TfidfVectorizer(
        max_features=20000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = tfidf.fit_transform(
        train_df["text_blob"]
    )

    X_test_tfidf = tfidf.transform(
        test_df["text_blob"]
    )

    print(
        f"TF-IDF training shape: "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"TF-IDF testing shape: "
        f"{X_test_tfidf.shape}"
    )

    # ==========================================================
    # 6. TRUNCATED SVD
    # ==========================================================

    log(
        f"Reducing TF-IDF dimensionality "
        f"to {SVD_COMPONENTS} components..."
    )

    svd = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        random_state=RANDOM_STATE,
    )

    X_train_svd = svd.fit_transform(X_train_tfidf)

    X_test_svd = svd.transform(X_test_tfidf)

    explained_variance = svd.explained_variance_ratio_.sum()

    print(
        f"SVD training shape: "
        f"{X_train_svd.shape}"
    )

    print(
        f"SVD testing shape: "
        f"{X_test_svd.shape}"
    )

    print(
        f"Explained variance ratio: "
        f"{explained_variance:.4f}"
    )

    # ==========================================================
    # 7. PROCESS RIASEC FEATURES
    # ==========================================================

    log("Processing RIASEC features...")

    scaler = StandardScaler()

    X_train_riasec = scaler.fit_transform(
        train_df[NUMERIC_COLS].values
    )

    X_test_riasec = scaler.transform(
        test_df[NUMERIC_COLS].values
    )

    print(
        f"RIASEC training shape: "
        f"{X_train_riasec.shape}"
    )

    # ==========================================================
    # 8. COMBINE FEATURES
    # ==========================================================

    log("Combining SVD + RIASEC features...")

    X_train = np.hstack(
        [
            X_train_svd,
            X_train_riasec,
        ]
    )

    X_test = np.hstack(
        [
            X_test_svd,
            X_test_riasec,
        ]
    )

    print(
        f"Final training feature shape: "
        f"{X_train.shape}"
    )

    print(
        f"Final testing feature shape: "
        f"{X_test.shape}"
    )

    # ==========================================================
    # 9. SAVE FEATURES
    # ==========================================================

    log("Saving feature matrices...")

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test,
    )

    # ==========================================================
    # 10. SAVE PREPROCESSING OBJECTS
    # ==========================================================

    log("Saving preprocessing objects...")

    joblib.dump(
        tfidf,
        OUTPUT_DIR / "tfidf_vectorizer.joblib",
    )

    joblib.dump(
        svd,
        OUTPUT_DIR / "svd.joblib",
    )

    joblib.dump(
        scaler,
        OUTPUT_DIR / "riasec_scaler.joblib",
    )

    joblib.dump(
        label_encoder,
        OUTPUT_DIR / "label_encoder.joblib",
    )

    # ==========================================================
    # 11. SAVE METADATA
    # ==========================================================

    metadata = {
        "dataset_rows": int(len(df)),
        "career_classes": int(len(label_encoder.classes_)),
        "training_samples": int(len(train_df)),
        "testing_samples": int(len(test_df)),
        "tfidf_features": 20000,
        "svd_components": SVD_COMPONENTS,
        "riasec_features": len(NUMERIC_COLS),
        "final_features": int(X_train.shape[1]),
        "svd_explained_variance": float(explained_variance),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "text_fields": [
            "Skills",
            "Education",
            "Experience",
        ],
        "excluded_fields": [
            "Job_Description",
            "Career",
            "Profile_ID",
        ],
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    # ==========================================================
    # COMPLETE
    # ==========================================================

    print()
    print("=" * 75)
    print("STEP 1 COMPLETE")
    print("=" * 75)

    print(
        f"Features created: {X_train.shape[1]}"
    )

    print(
        f"SVD explained variance: "
        f"{explained_variance:.4f}"
    )

    print()
    print("Saved files:")

    print(
        OUTPUT_DIR / "X_train.npy"
    )

    print(
        OUTPUT_DIR / "X_test.npy"
    )

    print(
        OUTPUT_DIR / "y_train.npy"
    )

    print(
        OUTPUT_DIR / "y_test.npy"
    )

    print(
        OUTPUT_DIR / "tfidf_vectorizer.joblib"
    )

    print(
        OUTPUT_DIR / "svd.joblib"
    )

    print(
        OUTPUT_DIR / "riasec_scaler.joblib"
    )

    print(
        OUTPUT_DIR / "label_encoder.joblib"
    )

    print(
        OUTPUT_DIR / "metadata.json"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
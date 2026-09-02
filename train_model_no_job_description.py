"""
CareerCast - Milestone 2
Leakage-resistant baseline ML training

Experiment B:
Uses:
    Skills
    Education
    Experience
    RIASEC features

IMPORTANT:
Job_Description is intentionally excluded because each career has
exactly one unique job description in the generated dataset.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
)

import joblib


# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH = Path("results/careercast_candidate_profiles.csv")

MODEL_DIR = Path("results/model_no_job_description")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TFIDF_PARAMS = dict(
    max_features=20000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

NUMERIC_COLS = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]

CATEGORICAL_COLS = [
    "Education",
    "Experience",
]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 75)
    print("CAREERCAST MILESTONE 2")
    print("LEAKAGE-RESISTANT BASELINE ML TRAINING")
    print("=" * 75)

    # ------------------------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------------------------

    log("Loading candidate profiles...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {DATA_PATH.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique careers: {df['Career'].nunique()}")

    missing = df.isnull().sum().sum()

    print(f"Total missing values: {missing}")

    if missing > 0:
        required_cols = (
            ["Career", "Skills"]
            + NUMERIC_COLS
            + CATEGORICAL_COLS
        )

        df = df.dropna(subset=required_cols)

        print(f"Rows after dropping missing: {len(df)}")

    # ------------------------------------------------------------
    # 2. CREATE TEXT FEATURES
    # ------------------------------------------------------------

    log("Creating leakage-resistant text features...")

    # IMPORTANT:
    # Job_Description is intentionally NOT included.

    df["text_blob"] = (
        df["Skills"].astype(str)
        + " . "
        + df["Education"].astype(str)
        + " . "
        + df["Experience"].astype(str)
    )

    print()
    print("Text fields used:")
    print(" - Skills")
    print(" - Education")
    print(" - Experience")
    print()
    print("Excluded:")
    print(" - Job_Description")
    print(" - Career")
    print(" - Profile_ID")

    # ------------------------------------------------------------
    # 3. ENCODE CAREER LABELS
    # ------------------------------------------------------------

    log("Encoding career labels...")

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["Career"])

    n_classes = len(label_encoder.classes_)

    print(f"Number of career classes: {n_classes}")

    # ------------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # ------------------------------------------------------------

    log("Creating stratified train/test split...")

    indices = np.arange(len(df))

    idx_train, idx_test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = df.iloc[idx_train].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)

    y_train = y[idx_train]
    y_test = y[idx_test]

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    # ------------------------------------------------------------
    # 5. TF-IDF
    # ------------------------------------------------------------

    log("Creating TF-IDF features...")

    tfidf = TfidfVectorizer(**TFIDF_PARAMS)

    X_train_text = tfidf.fit_transform(
        train_df["text_blob"]
    )

    X_test_text = tfidf.transform(
        test_df["text_blob"]
    )

    print(
        f"TF-IDF training shape: "
        f"{X_train_text.shape}"
    )

    print(
        f"TF-IDF testing shape: "
        f"{X_test_text.shape}"
    )

    # ------------------------------------------------------------
    # 6. RIASEC FEATURES
    # ------------------------------------------------------------

    log("Processing RIASEC numerical features...")

    scaler = StandardScaler(
        with_mean=False
    )

    X_train_num = scaler.fit_transform(
        train_df[NUMERIC_COLS].values
    )

    X_test_num = scaler.transform(
        test_df[NUMERIC_COLS].values
    )

    X_train_num_sparse = sparse.csr_matrix(
        X_train_num
    )

    X_test_num_sparse = sparse.csr_matrix(
        X_test_num
    )

    # ------------------------------------------------------------
    # 7. COMBINE FEATURES
    # ------------------------------------------------------------

    log("Combining TF-IDF + RIASEC features...")

    X_train = sparse.hstack(
        [
            X_train_text,
            X_train_num_sparse
        ],
        format="csr",
    )

    X_test = sparse.hstack(
        [
            X_test_text,
            X_test_num_sparse
        ],
        format="csr",
    )

    print(
        f"Final training feature shape: "
        f"{X_train.shape}"
    )

    print(
        f"Final testing feature shape: "
        f"{X_test.shape}"
    )

    # ------------------------------------------------------------
    # 8. TRAIN SGD CLASSIFIER
    # ------------------------------------------------------------

    log("Training leakage-resistant SGDClassifier...")

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=50,
        tol=1e-3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping=False,
        class_weight="balanced",
    )

    start_time = time.time()

    clf.fit(
        X_train,
        y_train
    )

    elapsed = time.time() - start_time

    print()
    log(
        f"Training completed in "
        f"{elapsed:.1f} seconds."
    )

    # ------------------------------------------------------------
    # 9. PREDICTION
    # ------------------------------------------------------------

    log("Generating predictions...")

    y_pred = clf.predict(X_test)

    y_proba = clf.predict_proba(X_test)

    # ------------------------------------------------------------
    # 10. EVALUATION
    # ------------------------------------------------------------

    log("Evaluating model...")

    top1 = accuracy_score(
        y_test,
        y_pred
    )

    top3 = top_k_accuracy_score(
        y_test,
        y_proba,
        k=3,
        labels=clf.classes_,
    )

    top5 = top_k_accuracy_score(
        y_test,
        y_proba,
        k=5,
        labels=clf.classes_,
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    print()
    print("=" * 75)
    print("LEAKAGE-RESISTANT BASELINE RESULTS")
    print("=" * 75)

    print(f"Number of classes : {n_classes}")
    print(f"Training samples  : {len(y_train)}")
    print(f"Testing samples   : {len(y_test)}")
    print(f"Features          : {X_train.shape[1]}")

    print()
    print(f"Top-1 Accuracy    : {top1:.4f}")
    print(f"Top-3 Accuracy    : {top3:.4f}")
    print(f"Top-5 Accuracy    : {top5:.4f}")

    print()

    print(
        "Macro avg -> "
        f"Precision: {report['macro avg']['precision']:.4f}  "
        f"Recall: {report['macro avg']['recall']:.4f}  "
        f"F1: {report['macro avg']['f1-score']:.4f}"
    )

    print(
        "Weighted avg -> "
        f"Precision: {report['weighted avg']['precision']:.4f}  "
        f"Recall: {report['weighted avg']['recall']:.4f}  "
        f"F1: {report['weighted avg']['f1-score']:.4f}"
    )

    # ------------------------------------------------------------
    # 11. SAVE ARTIFACTS
    # ------------------------------------------------------------

    log("Saving model artifacts...")

    joblib.dump(
        clf,
        MODEL_DIR / "career_model.joblib"
    )

    joblib.dump(
        tfidf,
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.joblib"
    )

    joblib.dump(
        label_encoder,
        MODEL_DIR / "label_encoder.joblib"
    )

    with open(
        MODEL_DIR / "classification_report.json",
        "w"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    metadata = {
        "experiment": "Leakage-resistant baseline",
        "job_description_used": False,
        "n_classes": n_classes,
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
        "n_features": int(X_train.shape[1]),
        "model_type": "SGDClassifier(loss='log_loss')",
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "top5_accuracy": float(top5),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "text_fields_used": [
            "Skills",
            "Education",
            "Experience"
        ],
        "numeric_fields_used": NUMERIC_COLS,
    }

    with open(
        MODEL_DIR / "metadata.json",
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print("=" * 75)
    print("EXPERIMENT B COMPLETE")
    print("=" * 75)

    print(
        f"Artifacts saved to: "
        f"{MODEL_DIR.resolve()}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
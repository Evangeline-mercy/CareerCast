"""
CareerCast - Milestone 2
Baseline ML career prediction model (fast, sparse-friendly version)

Fixes the "stuck / never finishes" problem caused by using
LogisticRegression(lbfgs, multinomial) on 878 classes x ~35k sparse
features. Replaces it with SGDClassifier(loss="log_loss"), which is
built for exactly this shape of problem: large sparse matrices, many
classes, linear model.
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
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
import joblib

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = Path("results/careercast_candidate_profiles.csv")
MODEL_DIR = Path("results/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TFIDF_PARAMS = dict(
    max_features=20000,   # cap dimensionality -> big speed win, minimal signal loss
    min_df=2,              # drop terms that appear in only 1 profile (likely noise)
    max_df=0.95,            # drop near-universal terms
    ngram_range=(1, 2),
    sublinear_tf=True,
)

NUMERIC_COLS = [
    "Realistic", "Investigative", "Artistic",
    "Social", "Enterprising", "Conventional",
]
# Education / Experience are categorical text ("Bachelor's degree", "2-4 years", etc.)
# -> we encode them as extra TF-IDF-able text merged into the text blob AND also
#    keep a lightweight ordinal/frequency numeric fallback below.
CATEGORICAL_COLS = ["Education", "Experience"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    print("=" * 70)
    print("CAREERCAST MILESTONE 2 - BASELINE ML TRAINING (fast/sparse version)")
    print("=" * 70)

    # ------------------------------------------------------------------
    log("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique careers: {df['Career'].nunique()}")

    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    if missing > 0:
        df = df.dropna(subset=["Career", "Skills", "Job_Description"] + NUMERIC_COLS + CATEGORICAL_COLS)
        print(f"Rows after dropping missing: {len(df)}")

    # ------------------------------------------------------------------
    log("Creating text features...")
    # Combine the genuinely textual, informative fields. Career/Profile_ID
    # are NEVER included here (that would be leakage / label-as-feature).
    df["text_blob"] = (
        df["Skills"].astype(str) + " . " +
        df["Job_Description"].astype(str) + " . " +
        df["Education"].astype(str) + " . " +
        df["Experience"].astype(str)
    )

    # ------------------------------------------------------------------
    log("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Career"])
    n_classes = len(label_encoder.classes_)

    # ------------------------------------------------------------------
    log("Splitting dataset (stratified)...")
    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
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

    # ------------------------------------------------------------------
    log("Creating TF-IDF features...")
    # IMPORTANT: fit ONLY on training text, transform test separately.
    # Fitting on the full dataset before splitting would leak test
    # vocabulary/idf statistics into training -> avoided here.
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_text = tfidf.fit_transform(train_df["text_blob"])
    X_test_text = tfidf.transform(test_df["text_blob"])
    print(f"TF-IDF training shape: {X_train_text.shape}")
    print(f"TF-IDF testing shape: {X_test_text.shape}")

    # ------------------------------------------------------------------
    log("Processing numerical features...")
    # RIASEC scores are already numeric -> scale them.
    scaler = StandardScaler(with_mean=False)  # with_mean=False keeps output sparse-friendly
    X_train_num = scaler.fit_transform(train_df[NUMERIC_COLS].values)
    X_test_num = scaler.transform(test_df[NUMERIC_COLS].values)

    X_train_num_sparse = sparse.csr_matrix(X_train_num)
    X_test_num_sparse = sparse.csr_matrix(X_test_num)

    # ------------------------------------------------------------------
    log("Combining TF-IDF + numerical features...")
    X_train = sparse.hstack([X_train_text, X_train_num_sparse], format="csr")
    X_test = sparse.hstack([X_test_text, X_test_num_sparse], format="csr")
    print(f"Final training feature shape: {X_train.shape}")
    print(f"Final testing feature shape: {X_test.shape}")

    # ------------------------------------------------------------------
    log("Training model (SGDClassifier, log_loss)...")
    log("This is designed to be fast even with many classes.")
    clf = SGDClassifier(
        loss="log_loss",       # gives predict_proba, needed for top-k + future recommendations
        penalty="l2",
        alpha=1e-5,
        max_iter=50,
        tol=1e-3,
        n_jobs=-1,              # parallel one-vs-rest style updates
        random_state=RANDOM_STATE,
        early_stopping=False,
        class_weight="balanced",  # helps since some careers may be marginally harder to separate
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    log(f"Training completed in {time.time() - t0:.1f} seconds.")

    # ------------------------------------------------------------------
    log("Evaluating model...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy_score(y_test, y_proba, k=3, labels=clf.classes_)
    top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=clf.classes_)

    print(f"\nNumber of classes: {n_classes}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"\nTop-1 Accuracy: {acc:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")

    # classification_report with 878 classes is huge -> only print a
    # condensed macro/weighted summary, and save the full report to file.
    report_dict = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    print("\nMacro avg  -> precision: {:.3f}  recall: {:.3f}  f1: {:.3f}".format(
        report_dict["macro avg"]["precision"],
        report_dict["macro avg"]["recall"],
        report_dict["macro avg"]["f1-score"],
    ))
    print("Weighted avg -> precision: {:.3f}  recall: {:.3f}  f1: {:.3f}".format(
        report_dict["weighted avg"]["precision"],
        report_dict["weighted avg"]["recall"],
        report_dict["weighted avg"]["f1-score"],
    ))

    with open(MODEL_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # ------------------------------------------------------------------
    log("Saving model and preprocessing objects...")
    joblib.dump(clf, MODEL_DIR / "career_model.joblib")
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

    metadata = {
        "n_classes": n_classes,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "tfidf_params": TFIDF_PARAMS,
        "numeric_cols": NUMERIC_COLS,
        "text_fields_used": ["Skills", "Job_Description", "Education", "Experience"],
        "model_type": "SGDClassifier(loss='log_loss')",
        "top1_accuracy": acc,
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"All artifacts saved to {MODEL_DIR.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
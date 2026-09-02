import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 90)

# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
    "results/milestone2_training/"
    "career_profile_training_dataset.csv"
)

OUTPUT_DIR = "results/milestone2_profile_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "logistic_metrics.csv"
)

PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "logistic_predictions.csv"
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "logistic_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    OUTPUT_DIR,
    "tfidf_vectorizer.pkl"
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1] Loading training dataset...")

df = pd.read_csv(INPUT_PATH)

print("Rows    :", len(df))
print("Careers :", df["career"].nunique())

# Remove empty records
df = df.dropna(subset=["skills", "career"]).copy()

X_text = df["skills"].astype(str)
y = df["career"].astype(str)

# ============================================================
# 2. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

print("\n[2] Creating stratified train/test split...")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples :", len(X_train_text))
print("Testing samples  :", len(X_test_text))

# ============================================================
# 3. TF-IDF
# ============================================================

print("\n[3] Building TF-IDF skill representation...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_features=None,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("TF-IDF training shape :", X_train.shape)
print("TF-IDF testing shape  :", X_test.shape)
print("Vocabulary size       :", len(vectorizer.vocabulary_))

# ============================================================
# 4. LOGISTIC REGRESSION
# ============================================================

print("\n[4] Training Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    C=5.0,
    solver="lbfgs",
    random_state=42
)

model.fit(X_train, y_train)

print("Logistic Regression training complete.")

# ============================================================
# 5. EVALUATION
# ============================================================

print("\n[5] Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 90)
print("LOGISTIC REGRESSION TEST RESULTS")
print("=" * 90)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Macro F1  : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 90)
print("CLASSIFICATION REPORT")
print("=" * 90)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

# ============================================================
# 7. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": macro_f1,
        "training_samples": len(X_train_text),
        "testing_samples": len(X_test_text),
        "careers": y.nunique(),
        "vocabulary_size": len(vectorizer.vocabulary_)
    }
])

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

# ============================================================
# 8. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "actual_career": y_test.values,
    "predicted_career": y_pred
})

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

# ============================================================
# 9. SAVE TRAINED MODEL
# ============================================================

print("\n[6] Saving trained model and TF-IDF vectorizer...")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

# ============================================================
# 10. VERIFY FILES
# ============================================================

print("\nSaved:")

print(METRICS_PATH)
print(PREDICTIONS_PATH)
print(MODEL_PATH)
print(VECTORIZER_PATH)

print("\nFile verification:")

print(
    "Logistic model     :",
    os.path.exists(MODEL_PATH)
)

print(
    "TF-IDF vectorizer  :",
    os.path.exists(VECTORIZER_PATH)
)

print("\n" + "=" * 90)
print("LOGISTIC REGRESSION BASELINE COMPLETE")
print("=" * 90)
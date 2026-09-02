from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "results" / "milestone2_validation"

MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "milestone2_sentence_bert_classifier_v2"
)

VALIDATION_PATH = DATA_DIR / "validation.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("CAREERCAST - MILESTONE 2 VALIDATION EVALUATION")
print("=" * 90)

print("\nLoading validation data...")

df = pd.read_csv(VALIDATION_PATH)

df["career"] = df["career"].astype(str)
df["skills"] = df["skills"].fillna("").astype(str)

print("Validation rows :", len(df))
print("Careers         :", df["career"].nunique())


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

lr = joblib.load(
    MODEL_DIR / "logistic_regression_model.pkl"
)

rf = joblib.load(
    MODEL_DIR / "random_forest_model.pkl"
)

xgb = joblib.load(
    MODEL_DIR / "xgboost_model.pkl"
)

label_encoder = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)

print("Models loaded.")


# ============================================================
# LOAD SBERT
# ============================================================

print("\nLoading SBERT...")

sbert = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# GENERATE VALIDATION EMBEDDINGS
# ============================================================

print("\nGenerating validation embeddings...")

X = sbert.encode(
    df["skills"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

X = np.asarray(X, dtype=np.float32)

print("Embedding shape:", X.shape)


# ============================================================
# TRUE LABELS
# ============================================================

y_true = label_encoder.transform(
    df["career"]
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model):

    print("\n" + "=" * 90)
    print(name.upper())
    print("=" * 90)

    probabilities = model.predict_proba(X)

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # TOP-1
    # --------------------------------------------------------

    top1 = accuracy_score(
        y_true,
        predictions
    )

    # --------------------------------------------------------
    # TOP-3 / TOP-5
    # --------------------------------------------------------

    sorted_ids = np.argsort(
        probabilities,
        axis=1
    )[:, ::-1]

    top3_predictions = sorted_ids[:, :3]
    top5_predictions = sorted_ids[:, :5]

    top3_correct = np.any(
        top3_predictions == y_true[:, None],
        axis=1
    )

    top5_correct = np.any(
        top5_predictions == y_true[:, None],
        axis=1
    )

    top3 = top3_correct.mean()
    top5 = top5_correct.mean()

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\nOVERALL METRICS")
    print("-" * 90)

    print(f"Top-1 Accuracy : {top1 * 100:.2f}%")
    print(f"Top-3 Accuracy : {top3 * 100:.2f}%")
    print(f"Top-5 Accuracy : {top5 * 100:.2f}%")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")

    # --------------------------------------------------------
    # PER-CAREER ACCURACY
    # --------------------------------------------------------

    results = []

    for class_id, career in enumerate(
        label_encoder.classes_
    ):

        mask = y_true == class_id

        if mask.sum() == 0:
            continue

        career_accuracy = (
            predictions[mask] == class_id
        ).mean()

        results.append(
            {
                "career": career,
                "samples": int(mask.sum()),
                "accuracy": career_accuracy,
            }
        )

    career_df = pd.DataFrame(results)

    career_df = career_df.sort_values(
        "accuracy"
    )

    print("\nLOWEST 15 CAREER ACCURACIES")
    print("-" * 90)

    for _, row in career_df.head(15).iterrows():

        print(
            f"{row['career']:<55} "
            f"{row['accuracy'] * 100:.2f}%"
        )

    print("\nHIGHEST 15 CAREER ACCURACIES")
    print("-" * 90)

    for _, row in career_df.tail(15).sort_values(
        "accuracy",
        ascending=False
    ).iterrows():

        print(
            f"{row['career']:<55} "
            f"{row['accuracy'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # SAVE PER-CAREER RESULTS
    # --------------------------------------------------------

    output_path = (
        MODEL_DIR
        / f"{name.lower().replace(' ', '_')}_career_accuracy.csv"
    )

    career_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved career accuracy:"
        f"\n{output_path}"
    )

    return {
        "model": name,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# RUN ALL THREE MODELS
# ============================================================

results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        lr
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        rf
    )
)

results.append(
    evaluate_model(
        "XGBoost",
        xgb
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(results)

print("\n" + "=" * 90)
print("FINAL VALIDATION SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        formatters={
            "top1": "{:.4f}".format,
            "top3": "{:.4f}".format,
            "top5": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
            "weighted_f1": "{:.4f}".format,
        }
    )
)

summary.to_csv(
    MODEL_DIR / "validation_summary.csv",
    index=False
)

print("\nSaved:")
print(
    MODEL_DIR / "validation_summary.csv"
)

print("\nVALIDATION EVALUATION COMPLETED.")
"""
CareerCast - Milestone 2
Final Test-Set Evaluation

Uses:
- results/milestone2_validation/test.csv
- results/milestone2_sentence_bert_classifier_v2/
- all-MiniLM-L6-v2
- Logistic Regression
- Random Forest
- XGBoost

The test set is NEVER used for training.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "milestone2_validation"
    / "test.csv"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "results"
    / "milestone2_sentence_bert_classifier_v2"
)

LR_PATH = ARTIFACT_DIR / "logistic_regression_model.pkl"
RF_PATH = ARTIFACT_DIR / "random_forest_model.pkl"
XGB_PATH = ARTIFACT_DIR / "xgboost_model.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 90)
print("CAREERCAST - MILESTONE 2 FINAL TEST EVALUATION")
print("=" * 90)

print("\nLoading test data...")

df = pd.read_csv(TEST_PATH)

df["skills"] = df["skills"].fillna("")

texts = df["skills"].astype(str).tolist()
true_careers = df["career"].astype(str).tolist()

print(f"Test rows : {len(df)}")
print(f"Careers   : {df['career'].nunique()}")

# Verify expected structure
counts = df["career"].value_counts()

print(f"Minimum samples/career : {counts.min()}")
print(f"Maximum samples/career : {counts.max()}")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

lr = joblib.load(LR_PATH)
rf = joblib.load(RF_PATH)
xgb = joblib.load(XGB_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Models loaded.")

print(f"LR classes  : {len(lr.classes_)}")
print(f"RF classes  : {len(rf.classes_)}")
print(f"XGB classes : {len(xgb.classes_)}")


# ============================================================
# LOAD SBERT
# ============================================================

print("\nLoading SBERT...")
print(SBERT_MODEL_NAME)

sbert = SentenceTransformer(SBERT_MODEL_NAME)


# ============================================================
# GENERATE TEST EMBEDDINGS
# ============================================================

print("\nGenerating test embeddings...")

X_test = sbert.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

X_test = np.asarray(
    X_test,
    dtype=np.float32,
)

print(f"Embedding shape: {X_test.shape}")

print(
    f"Average embedding norm: "
    f"{np.linalg.norm(X_test, axis=1).mean():.4f}"
)


# ============================================================
# TRUE LABEL IDS
# ============================================================

y_test = label_encoder.transform(
    true_careers
)


# ============================================================
# TOP-K ACCURACY FUNCTION
# ============================================================

def top_k_accuracy(model, X, y, k):

    probabilities = model.predict_proba(X)

    top_k_indices = np.argsort(
        probabilities,
        axis=1
    )[:, -k:]

    correct = np.any(
        top_k_indices == y.reshape(-1, 1),
        axis=1,
    )

    return correct.mean()


# ============================================================
# CAREER-LEVEL ACCURACY
# ============================================================

def career_accuracy(model, X, careers):

    probabilities = model.predict_proba(X)

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    predicted_careers = label_encoder.inverse_transform(
        predictions
    )

    result = {}

    for career in sorted(set(careers)):

        indices = [
            i
            for i, c in enumerate(careers)
            if c == career
        ]

        correct = sum(
            predicted_careers[i] == career
            for i in indices
        )

        result[career] = correct / len(indices)

    return result


# ============================================================
# MODEL EVALUATION
# ============================================================

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb,
}

summary = []


for model_name, model in models.items():

    print("\n" + "=" * 90)
    print(model_name.upper())
    print("=" * 90)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Top-1
    # --------------------------------------------------------

    top1 = accuracy_score(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Top-3
    # --------------------------------------------------------

    top3 = top_k_accuracy(
        model,
        X_test,
        y_test,
        3,
    )

    # --------------------------------------------------------
    # Top-5
    # --------------------------------------------------------

    top5 = top_k_accuracy(
        model,
        X_test,
        y_test,
        5,
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    print("\nOVERALL TEST METRICS")
    print("-" * 90)

    print(f"Top-1 Accuracy : {top1 * 100:.2f}%")
    print(f"Top-3 Accuracy : {top3 * 100:.2f}%")
    print(f"Top-5 Accuracy : {top5 * 100:.2f}%")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")

    # --------------------------------------------------------
    # Career-level accuracy
    # --------------------------------------------------------

    career_acc = career_accuracy(
        model,
        X_test,
        true_careers,
    )

    career_df = (
        pd.DataFrame(
            career_acc.items(),
            columns=[
                "career",
                "accuracy",
            ],
        )
        .sort_values(
            "accuracy",
            ascending=True,
        )
    )

    print("\nLOWEST 15 CAREER ACCURACIES")
    print("-" * 90)

    for _, row in career_df.head(15).iterrows():

        print(
            f"{row['career']:<55}"
            f"{row['accuracy'] * 100:>7.2f}%"
        )

    print("\nHIGHEST 15 CAREER ACCURACIES")
    print("-" * 90)

    for _, row in career_df.tail(15).sort_values(
        "accuracy",
        ascending=False,
    ).iterrows():

        print(
            f"{row['career']:<55}"
            f"{row['accuracy'] * 100:>7.2f}%"
        )

    # --------------------------------------------------------
    # Save career accuracy
    # --------------------------------------------------------

    output_path = (
        ARTIFACT_DIR
        / (
            model_name.lower()
            .replace(" ", "_")
            + "_test_career_accuracy.csv"
        )
    )

    career_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved career accuracy:"
    )

    print(output_path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary.append(
        {
            "model": model_name,
            "top1": top1,
            "top3": top3,
            "top5": top5,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 90)
print("FINAL TEST SUMMARY")
print("=" * 90)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "top1": "{:.4f}".format,
            "top3": "{:.4f}".format,
            "top5": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
            "weighted_f1": "{:.4f}".format,
        },
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = summary_df.loc[
    summary_df["top1"].idxmax()
]

print("\n" + "=" * 90)
print("BEST MODEL")
print("=" * 90)

print(
    f"Model          : {best_model['model']}"
)

print(
    f"Top-1 Accuracy : "
    f"{best_model['top1'] * 100:.2f}%"
)

print(
    f"Top-3 Accuracy : "
    f"{best_model['top3'] * 100:.2f}%"
)

print(
    f"Top-5 Accuracy : "
    f"{best_model['top5'] * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{best_model['macro_f1']:.4f}"
)

# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_path = (
    ARTIFACT_DIR
    / "test_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)

print("\nSaved:")
print(summary_path)

print("\n" + "=" * 90)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 90)

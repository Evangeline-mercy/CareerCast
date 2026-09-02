from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "results" / "milestone2_validation"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "milestone2_sentence_bert_classifier_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = DATA_DIR / "train.csv"

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 90)
print("CAREERCAST - MILESTONE 2 SBERT MODEL RETRAINING")
print("=" * 90)

print("\nLoading training data...")

df = pd.read_csv(TRAIN_PATH)

df["career"] = df["career"].astype(str)
df["skills"] = df["skills"].fillna("").astype(str)

print(f"Training rows : {len(df)}")
print(f"Careers       : {df['career'].nunique()}")


# ============================================================
# LABEL ENCODING
# ============================================================

print("\nCreating label encoder...")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df["career"])

print(f"Number of classes: {len(label_encoder.classes_)}")

if len(label_encoder.classes_) != 96:
    raise ValueError(
        f"Expected 96 careers, found {len(label_encoder.classes_)}"
    )


# ============================================================
# LOAD SBERT
# ============================================================

print("\nLoading SBERT:")
print(SBERT_MODEL_NAME)

sbert_model = SentenceTransformer(SBERT_MODEL_NAME)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating 384-dimensional SBERT embeddings...")

X = sbert_model.encode(
    df["skills"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

X = np.asarray(X, dtype=np.float32)

print("\nEmbedding shape:", X.shape)

if X.shape != (len(df), 384):
    raise ValueError(
        f"Unexpected embedding shape: {X.shape}"
    )


# ============================================================
# VERIFY NORMALIZATION
# ============================================================

norms = np.linalg.norm(X, axis=1)

print(
    "Average embedding norm:",
    float(norms.mean())
)

if not np.allclose(norms, 1.0, atol=1e-4):
    raise ValueError(
        "Embeddings are not properly normalized."
    )


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 90)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 90)

lr = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
)

lr.fit(X, y)

print("Logistic Regression training completed.")


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 90)
print("TRAINING RANDOM FOREST")
print("=" * 90)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X, y)

print("Random Forest training completed.")


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 90)
print("TRAINING XGBOOST")
print("=" * 90)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=96,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

xgb.fit(X, y)

print("XGBoost training completed.")


# ============================================================
# VERIFY CLASS IDS
# ============================================================

print("\n" + "=" * 90)
print("VERIFYING MODEL CLASS IDS")
print("=" * 90)

expected_classes = np.arange(96)

for name, model in [
    ("Logistic Regression", lr),
    ("Random Forest", rf),
    ("XGBoost", xgb),
]:

    print(
        f"{name}:",
        len(model.classes_),
        "classes"
    )

    if not np.array_equal(
        model.classes_,
        expected_classes
    ):
        raise ValueError(
            f"{name} class IDs are incompatible."
        )

    print(f"{name}: class IDs VERIFIED")


# ============================================================
# SAVE MODELS
# ============================================================

print("\n" + "=" * 90)
print("SAVING ARTIFACTS")
print("=" * 90)

joblib.dump(
    lr,
    OUTPUT_DIR / "logistic_regression_model.pkl"
)

joblib.dump(
    rf,
    OUTPUT_DIR / "random_forest_model.pkl"
)

joblib.dump(
    xgb,
    OUTPUT_DIR / "xgboost_model.pkl"
)

joblib.dump(
    label_encoder,
    OUTPUT_DIR / "label_encoder.pkl"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "sbert_model": SBERT_MODEL_NAME,
    "embedding_dimension": 384,
    "num_classes": 96,
    "training_rows": len(df),
    "models": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ],
}

joblib.dump(
    metadata,
    OUTPUT_DIR / "metadata.pkl"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 90)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 90)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSaved files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nEmbedding dimension :", X.shape[1])
print("Training samples    :", X.shape[0])
print("Career classes      :", len(label_encoder.classes_))

print("\nNEXT STEP:")
print("Evaluate the new models using validation.csv and test.csv.")
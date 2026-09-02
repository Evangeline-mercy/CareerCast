import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

INPUT = Path(
    "results/milestone2_unified/milestone2_training_dataset.csv"
)

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("RANDOM FOREST BASELINE")
print("=" * 90)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading training dataset...")

df = pd.read_csv(INPUT)

df["skills"] = df["skills"].fillna("").astype(str)
df["career"] = df["career"].astype(str).str.strip()

print("Rows    :", len(df))
print("Careers :", df["career"].nunique())

# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X_text = df["skills"]
y = df["career"]

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
    token_pattern=r"(?u)\b[\w.+#-]+\b",
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("TF-IDF training shape :", X_train.shape)
print("TF-IDF testing shape  :", X_test.shape)
print("Vocabulary size       :", len(vectorizer.vocabulary_))

# ============================================================
# 4. RANDOM FOREST
# ============================================================

print("\n[4] Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Random Forest training complete.")

# ============================================================
# 5. PREDICTION
# ============================================================

print("\n[5] Evaluating Random Forest...")

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

# ============================================================
# 6. RESULTS
# ============================================================

print("\n" + "=" * 90)
print("RANDOM FOREST TEST RESULTS")
print("=" * 90)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Macro F1  : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

print("\n" + "=" * 90)
print("CLASSIFICATION REPORT")
print("=" * 90)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================================
# 7. TOP FEATURES
# ============================================================

print("\n" + "=" * 90)
print("TOP GLOBAL TF-IDF FEATURES")
print("=" * 90)

feature_names = np.array(vectorizer.get_feature_names_out())

importance = rf.feature_importances_

top_indices = np.argsort(importance)[::-1][:30]

for i in top_indices:
    print(
        f"{feature_names[i]:<30} "
        f"{importance[i]:.6f}"
    )

# ============================================================
# 8. SAVE MODEL INFORMATION
# ============================================================

OUTPUT_DIR = Path("results/milestone2_rf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metrics = pd.DataFrame([{
    "model": "Random Forest",
    "accuracy": accuracy,
    "precision_weighted": precision,
    "recall_weighted": recall,
    "macro_f1": macro_f1,
    "train_samples": len(X_train_text),
    "test_samples": len(X_test_text),
    "num_careers": y.nunique()
}])

metrics.to_csv(
    OUTPUT_DIR / "random_forest_metrics.csv",
    index=False
)

predictions = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred
})

predictions.to_csv(
    OUTPUT_DIR / "random_forest_predictions.csv",
    index=False
)

print("\nSaved:")
print(OUTPUT_DIR / "random_forest_metrics.csv")
print(OUTPUT_DIR / "random_forest_predictions.csv")

print("\n" + "=" * 90)
print("RANDOM FOREST BASELINE COMPLETE")
print("=" * 90)
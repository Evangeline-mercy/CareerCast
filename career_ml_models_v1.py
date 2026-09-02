# ================================================================
# CAREERCAST MILESTONE 2
# RANDOM FOREST + XGBOOST
# CROSS-VALIDATED HYPERPARAMETER TUNING
# ================================================================

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

DATASET_PATH = "datasets/global_ai_jobs_dataset.csv"

OUTPUT_ROOT = "v4_output"

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

# Limit TF-IDF vocabulary to keep training practical
MAX_FEATURES = 5000


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("RANDOM FOREST + XGBOOST")
print("CROSS-VALIDATED HYPERPARAMETER TUNING")
print("=" * 80)


# ================================================================
# 1. LOAD DATASET
# ================================================================

print("\n" + "=" * 80)
print("1. LOADING DATASET")
print("=" * 80)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

required_columns = [
    "job_title",
    "skills",
    "tools_used"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ================================================================
# 2. CLEAN TARGET
# ================================================================

print("\n" + "=" * 80)
print("2. PREPARING TARGET")
print("=" * 80)

df["job_title"] = (
    df["job_title"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df["skills"] = (
    df["skills"]
    .fillna("")
    .astype(str)
)

df["tools_used"] = (
    df["tools_used"]
    .fillna("")
    .astype(str)
)

# Remove rows without target
df = df[df["job_title"] != ""].copy()

print(f"Valid rows: {len(df)}")

print("\nCareer classes:")

career_counts = df["job_title"].value_counts()

for career, count in career_counts.items():
    print(f" - {career}: {count}")


# ================================================================
# 3. BUILD TEXT FEATURES
# ================================================================

print("\n" + "=" * 80)
print("3. BUILDING TF-IDF FEATURES")
print("=" * 80)

# Combine skills and tools.
#
# Skills are repeated twice intentionally so that skill information
# has more influence than tools in the first baseline model.

df["combined_text"] = (
    df["skills"] + " " +
    df["skills"] + " " +
    df["tools_used"]
)

df["combined_text"] = (
    df["combined_text"]
    .str.lower()
    .str.replace(",", " ", regex=False)
    .str.replace(";", " ", regex=False)
    .str.replace("|", " ", regex=False)
)

print("Text feature construction: COMPLETE")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.98,
    max_features=MAX_FEATURES,
    sublinear_tf=True
)

X = vectorizer.fit_transform(
    df["combined_text"]
)

y_text = df["job_title"]

print(f"TF-IDF matrix shape: {X.shape}")
print(f"TF-IDF vocabulary  : {len(vectorizer.vocabulary_)}")


# ================================================================
# 4. ENCODE TARGET LABELS
# ================================================================

print("\n" + "=" * 80)
print("4. ENCODING CAREER LABELS")
print("=" * 80)

careers = sorted(y_text.unique())

career_to_id = {
    career: i
    for i, career in enumerate(careers)
}

id_to_career = {
    i: career
    for career, i in career_to_id.items()
}

y = y_text.map(career_to_id).values

print(f"Number of career classes: {len(careers)}")

for i, career in id_to_career.items():
    print(f" {i}: {career}")


# ================================================================
# 5. TRAIN / TEST SPLIT
# ================================================================

print("\n" + "=" * 80)
print("5. TRAIN / TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ================================================================
# 6. CROSS VALIDATION
# ================================================================

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ================================================================
# 7. RANDOM FOREST
# ================================================================

print("\n" + "=" * 80)
print("6. RANDOM FOREST")
print("=" * 80)

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=8,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)

print("Starting Random Forest hyperparameter tuning...")
print(f"Cross-validation folds: {CV_FOLDS}")

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("\nRandom Forest tuning complete.")

print("\nBest Parameters:")
for key, value in rf_search.best_params_.items():
    print(f" - {key}: {value}")

print(
    f"\nBest CV Macro-F1: "
    f"{rf_search.best_score_:.4f}"
)


# ================================================================
# 8. RANDOM FOREST TEST EVALUATION
# ================================================================

rf_predictions = best_rf.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    average="macro",
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    average="macro",
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    average="macro",
    zero_division=0
)

print("\nRandom Forest Test Results:")
print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"Macro F1 : {rf_f1:.4f}")


# ================================================================
# 9. XGBOOST
# ================================================================

print("\n" + "=" * 80)
print("7. XGBOOST")
print("=" * 80)

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(careers),
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=xgb_params,
    n_iter=8,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)

print("Starting XGBoost hyperparameter tuning...")
print(f"Cross-validation folds: {CV_FOLDS}")

xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_

print("\nXGBoost tuning complete.")

print("\nBest Parameters:")
for key, value in xgb_search.best_params_.items():
    print(f" - {key}: {value}")

print(
    f"\nBest CV Macro-F1: "
    f"{xgb_search.best_score_:.4f}"
)


# ================================================================
# 10. XGBOOST TEST EVALUATION
# ================================================================

xgb_predictions = best_xgb.predict(X_test)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

xgb_precision = precision_score(
    y_test,
    xgb_predictions,
    average="macro",
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_predictions,
    average="macro",
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_predictions,
    average="macro",
    zero_division=0
)

print("\nXGBoost Test Results:")
print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall   : {xgb_recall:.4f}")
print(f"Macro F1 : {xgb_f1:.4f}")


# ================================================================
# 11. MODEL COMPARISON
# ================================================================

print("\n" + "=" * 80)
print("8. MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame([
    {
        "model": "Random Forest",
        "cv_macro_f1": rf_search.best_score_,
        "test_accuracy": rf_accuracy,
        "test_precision": rf_precision,
        "test_recall": rf_recall,
        "test_macro_f1": rf_f1
    },
    {
        "model": "XGBoost",
        "cv_macro_f1": xgb_search.best_score_,
        "test_accuracy": xgb_accuracy,
        "test_precision": xgb_precision,
        "test_recall": xgb_recall,
        "test_macro_f1": xgb_f1
    }
])

print(
    comparison.to_string(index=False)
)

best_model_name = comparison.loc[
    comparison["test_macro_f1"].idxmax(),
    "model"
]

print(f"\nBest Model: {best_model_name}")


# ================================================================
# 12. CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 80)
print("9. BEST MODEL CLASSIFICATION REPORT")
print("=" * 80)

if best_model_name == "Random Forest":
    best_predictions = rf_predictions
else:
    best_predictions = xgb_predictions

target_names = [
    id_to_career[i]
    for i in range(len(careers))
]

report = classification_report(
    y_test,
    best_predictions,
    target_names=target_names,
    zero_division=0
)

print(report)


# ================================================================
# 13. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ================================================================
# 14. SAVE OUTPUT
# ================================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_dir = os.path.join(
    OUTPUT_ROOT,
    f"ml_models_v1_{timestamp}"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# Save comparison
comparison_path = os.path.join(
    output_dir,
    "model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# Save classification report
report_path = os.path.join(
    output_dir,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# Save confusion matrix
cm_path = os.path.join(
    output_dir,
    "confusion_matrix.csv"
)

pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names
).to_csv(cm_path)


# Save metadata
metadata = {
    "dataset": DATASET_PATH,
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "career_classes": careers,
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "tfidf_features": int(X.shape[1]),
    "cv_folds": CV_FOLDS,
    "random_state": RANDOM_STATE,
    "random_forest": {
        "best_parameters": rf_search.best_params_,
        "cv_macro_f1": float(rf_search.best_score_),
        "test_accuracy": float(rf_accuracy),
        "test_precision": float(rf_precision),
        "test_recall": float(rf_recall),
        "test_macro_f1": float(rf_f1)
    },
    "xgboost": {
        "best_parameters": xgb_search.best_params_,
        "cv_macro_f1": float(xgb_search.best_score_),
        "test_accuracy": float(xgb_accuracy),
        "test_precision": float(xgb_precision),
        "test_recall": float(xgb_recall),
        "test_macro_f1": float(xgb_f1)
    },
    "best_model": best_model_name
}

metadata_path = os.path.join(
    output_dir,
    "model_results.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )


# ================================================================
# 15. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("10. OUTPUT FILES")
print("=" * 80)

print(f"\nOutput directory:")
print(os.path.abspath(output_dir))

print("\nFiles:")
print(f" - {comparison_path}")
print(f" - {report_path}")
print(f" - {cm_path}")
print(f" - {metadata_path}")

print("\n" + "=" * 80)
print("MILESTONE 2 REQUIREMENT 1 COMPLETE")
print("=" * 80)

print("\nRandom Forest:")
print(f" - CV Macro-F1 : {rf_search.best_score_:.4f}")
print(f" - Test Accuracy: {rf_accuracy:.4f}")

print("\nXGBoost:")
print(f" - CV Macro-F1 : {xgb_search.best_score_:.4f}")
print(f" - Test Accuracy: {xgb_accuracy:.4f}")

print(f"\nBest Model: {best_model_name}")

print("\nDataset: NOT MODIFIED")
print("V5 recommendation engine: NOT MODIFIED")
print("SBERT: NOT MODIFIED")
print("Benchmark validation: NOT MODIFIED")

print("\nMilestone 2 ML classifier training completed.")
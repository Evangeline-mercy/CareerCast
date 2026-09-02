import json
import os
import sys
import time
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20

INPUT_PATH = os.path.join("results", "milestone2_training", "career_profile_training_dataset.csv")
OUTPUT_DIR = os.path.join("results", "milestone2_random_forest")

SVD_COMPONENTS = 100
CV_FOLDS = 2

PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [None, 20],
    "max_features": ["sqrt"],
}

SCRIPT_START = time.time()


def log(msg):
    elapsed = time.time() - SCRIPT_START
    ts = datetime.now().strftime("%H:%M:%S")
    print("[" + ts + "] (+" + format(elapsed, ".1f") + "s) " + msg)


def fail(msg, exc=None):
    print()
    print("=" * 70)
    print("ERROR: " + msg)
    if exc is not None:
        print("-" * 70)
        traceback.print_exc()
    print("=" * 70)
    sys.exit(1)


def main():
    print("=" * 70)
    print("CAREERCAST MILESTONE 2 - RANDOM FOREST (profile_training track)")
    print("=" * 70)

    log("Loading dataset...")
    try:
        if not os.path.isfile(INPUT_PATH):
            fail("Dataset not found at '" + INPUT_PATH + "'.")

        df = pd.read_csv(INPUT_PATH)

        required_cols = {"skills", "career"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            fail("Dataset is missing required columns: " + str(sorted(missing_cols)))

        df = df.dropna(subset=["skills", "career"]).reset_index(drop=True)

        if len(df) == 0:
            fail("Dataset has 0 usable rows after dropping missing values.")
    except SystemExit:
        raise
    except Exception as e:
        fail("Failed to load dataset: " + str(e), e)

    n_careers = df["career"].nunique()
    log("Dataset loaded.")
    print("Dataset size: " + str(len(df)) + " rows")
    print("Number of career classes: " + str(n_careers))

    X_text = df["skills"].astype(str)
    y = df["career"].astype(str)

    log("Performing stratified train/test split...")
    try:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    except Exception as e:
        fail("Train/test split failed (check class balance): " + str(e), e)

    print("Training samples: " + str(len(X_train_text)))
    print("Testing samples: " + str(len(X_test_text)))

    log("Fitting TF-IDF vectorizer...")
    try:
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    except Exception as e:
        fail("TF-IDF vectorization failed: " + str(e), e)

    vocab_size = len(tfidf_vectorizer.vocabulary_)
    log("TF-IDF vectorization complete.")
    print("TF-IDF vocabulary size: " + str(vocab_size))
    print("TF-IDF train shape: " + str(X_train_tfidf.shape))
    print("TF-IDF test shape: " + str(X_test_tfidf.shape))

    log("Reducing dimensionality with TruncatedSVD (" + str(SVD_COMPONENTS) + " components)...")
    try:
        svd_components = min(SVD_COMPONENTS, min(X_train_tfidf.shape) - 1)
        svd = TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)
        X_train_svd = svd.fit_transform(X_train_tfidf).astype(np.float32, copy=False)
        X_test_svd = svd.transform(X_test_tfidf).astype(np.float32, copy=False)
        explained_variance = float(svd.explained_variance_ratio_.sum())
    except Exception as e:
        fail("TruncatedSVD failed: " + str(e), e)

    log("SVD complete.")
    print("SVD dimensions: " + str(X_train_svd.shape[1]))
    print("SVD explained variance (sum): " + format(explained_variance, ".4f"))

    total_fits = 1
    for v in PARAM_GRID.values():
        total_fits = total_fits * len(v)
    total_fits = total_fits * CV_FOLDS

    log("Setting up GridSearchCV (" + str(CV_FOLDS) + "-fold, " + str(total_fits) + " total fits)...")
    try:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        base_rf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=PARAM_GRID,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            verbose=2,
            refit=False,
        )
    except Exception as e:
        fail("Failed to set up GridSearchCV: " + str(e), e)

    log("Starting hyperparameter search...")
    try:
        search_start = time.time()
        grid_search.fit(X_train_svd, y_train)
        search_time = time.time() - search_start
    except Exception as e:
        fail("Hyperparameter search failed: " + str(e), e)

    best_params = grid_search.best_params_
    cv_accuracy = grid_search.best_score_
    log("Hyperparameter search complete. (" + format(search_time, ".1f") + "s)")
    print("Best Random Forest parameters: " + str(best_params))
    print("Cross-validation accuracy: " + format(cv_accuracy, ".4f"))

    log("Training final Random Forest model on full training set...")
    try:
        final_model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params,
        )
        train_start = time.time()
        final_model.fit(X_train_svd, y_train)
        training_time = time.time() - train_start
    except Exception as e:
        fail("Final model training failed: " + str(e), e)
    log("Training completed in " + format(training_time, ".1f") + " seconds.")

    log("Evaluating on test set...")
    try:
        y_pred = final_model.predict(X_test_svd)

        test_accuracy = accuracy_score(y_test, y_pred)
        precision_macro, recall_macro, macro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        report_text = classification_report(y_test, y_pred, zero_division=0)
    except Exception as e:
        fail("Evaluation failed: " + str(e), e)

    print()
    print("Test accuracy: " + format(test_accuracy, ".4f"))
    print("Macro precision: " + format(precision_macro, ".4f"))
    print("Macro recall: " + format(recall_macro, ".4f"))
    print("Macro F1: " + format(macro_f1, ".4f"))
    print()
    print("Classification report:")
    print(report_text)

    log("Saving artifacts...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        model_path = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")
        tfidf_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
        svd_path = os.path.join(OUTPUT_DIR, "svd_transformer.pkl")
        metrics_path = os.path.join(OUTPUT_DIR, "random_forest_metrics.csv")
        predictions_path = os.path.join(OUTPUT_DIR, "random_forest_predictions.csv")

        joblib.dump(final_model, model_path)
        joblib.dump(tfidf_vectorizer, tfidf_path)
        joblib.dump(svd, svd_path)

        metrics_row = {
            "model": "RandomForestClassifier",
            "accuracy": test_accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "macro_f1": macro_f1,
            "cv_accuracy": cv_accuracy,
            "training_samples": len(X_train_text),
            "testing_samples": len(X_test_text),
            "careers": n_careers,
            "vocabulary_size": vocab_size,
            "svd_components": X_train_svd.shape[1],
            "best_params": json.dumps(best_params),
        }
        pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)

        predictions_df = pd.DataFrame({
            "skills": X_test_text.reset_index(drop=True),
            "true_career": pd.Series(y_test).reset_index(drop=True),
            "predicted_career": y_pred,
        })
        predictions_df.to_csv(predictions_path, index=False)
    except Exception as e:
        fail("Failed while saving artifacts: " + str(e), e)

    log("Verifying saved files...")
    expected_files = [model_path, tfidf_path, svd_path, metrics_path, predictions_path]
    missing_after_save = [p for p in expected_files if not os.path.isfile(p)]
    if missing_after_save:
        fail("The following expected output files were not found after saving: " + str(missing_after_save))

    log("All output files verified.")
    for p in expected_files:
        print("  " + p)

    total_time = time.time() - SCRIPT_START
    print()
    print("Total time: " + format(total_time, ".1f") + " seconds")

    print()
    print("=" * 70)
    print("RANDOM FOREST TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail("Unhandled error in main().", e)
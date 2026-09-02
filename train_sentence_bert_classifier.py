import json
import os
import sys
import time
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost is not installed or not importable in this environment.")
    print(
        'Verify with: python -c "import xgboost; print(xgboost.__version__)"'
    )
    sys.exit(1)


# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_STATE = 42
CV_FOLDS = 2

DATASET_PATH = os.path.join(
    "results",
    "milestone2_training",
    "career_profile_training_dataset.csv",
)

EMBEDDINGS_DIR = os.path.join(
    "results",
    "milestone2_sentence_bert",
)

EMBEDDINGS_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "sentence_bert_embeddings.npy",
)

SPLIT_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "train_test_split_indices.npz",
)

MODEL_INFO_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "sentence_bert_model_info.json",
)

OUTPUT_DIR = os.path.join(
    "results",
    "milestone2_sentence_bert_classifier",
)


# ================================================================
# HYPERPARAMETER GRIDS
# ================================================================

# Logistic Regression
LOGREG_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
}


# Random Forest
RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [None, 20],
}


# XGBoost
XGB_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
}


XGB_FIXED_PARAMS = {
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


SCRIPT_START = time.time()


# ================================================================
# LOGGING
# ================================================================

def log(msg):
    elapsed = time.time() - SCRIPT_START
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] (+{elapsed:.1f}s) {msg}")


def fail(msg, exc=None):
    print()
    print("=" * 70)
    print("ERROR: " + msg)

    if exc is not None:
        print("-" * 70)
        traceback.print_exc()

    print("=" * 70)
    sys.exit(1)


# ================================================================
# MODEL EVALUATION FUNCTION
# ================================================================

def evaluate_model(
    name,
    estimator,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Perform hyperparameter tuning using ONLY the training data.

    IMPORTANT:
    The test set is NOT used during GridSearchCV.

    After selecting the best parameters using cross-validation,
    the final model is trained on the complete training set and
    evaluated once on the untouched test set.
    """

    # ------------------------------------------------------------
    # Calculate number of CV fits
    # ------------------------------------------------------------

    total_candidates = 1

    for values in param_grid.values():
        total_candidates *= len(values)

    total_fits = total_candidates * CV_FOLDS

    log(
        f"[{name}] Starting {CV_FOLDS}-fold GridSearchCV "
        f"({total_fits} total fits)..."
    )

    # ------------------------------------------------------------
    # Grid Search
    # ------------------------------------------------------------

    try:
        cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            verbose=1,

            # IMPORTANT:
            # Do not train the final model automatically here.
            refit=False,
        )

        search_start = time.time()

        # ONLY training data is used here.
        search.fit(X_train, y_train)

        search_time = time.time() - search_start

    except Exception as e:
        fail(
            f"[{name}] Hyperparameter search failed: {e}",
            e,
        )

    # ------------------------------------------------------------
    # Get best parameters from CV
    # ------------------------------------------------------------

    best_params = search.best_params_

    cv_accuracy = float(search.best_score_)

    log(
        f"[{name}] Search complete in {search_time:.1f}s."
    )

    print(
        f"[{name}] Best parameters: {best_params}"
    )

    print(
        f"[{name}] Cross-validation accuracy: "
        f"{cv_accuracy:.4f}"
    )

    # ------------------------------------------------------------
    # Train final model
    # ------------------------------------------------------------

    log(
        f"[{name}] Training final model on full training set..."
    )

    try:
        # Clone estimator parameters safely.
        final_params = estimator.get_params()

        # Replace tuned parameters with best parameters.
        final_params.update(best_params)

        final_estimator = estimator.__class__(
            **final_params
        )

        train_start = time.time()

        final_estimator.fit(
            X_train,
            y_train,
        )

        training_time = time.time() - train_start

    except Exception as e:
        fail(
            f"[{name}] Final model training failed: {e}",
            e,
        )

    log(
        f"[{name}] Training completed in "
        f"{training_time:.1f}s."
    )

    # ------------------------------------------------------------
    # FINAL TEST EVALUATION
    # ------------------------------------------------------------

    log(
        f"[{name}] Evaluating on untouched test set..."
    )

    try:
        y_pred = final_estimator.predict(X_test)

        test_accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        (
            precision_macro,
            recall_macro,
            macro_f1,
            _,
        ) = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

    except Exception as e:
        fail(
            f"[{name}] Evaluation failed: {e}",
            e,
        )

    log(
        f"[{name}] "
        f"CV accuracy={cv_accuracy:.4f} | "
        f"Test accuracy={test_accuracy:.4f} | "
        f"Macro F1={macro_f1:.4f}"
    )

    # ------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------

    return {
        "name": name,
        "model": final_estimator,
        "best_params": best_params,

        # Model selection metric
        "cv_accuracy": cv_accuracy,

        # Final held-out evaluation
        "test_accuracy": float(test_accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(macro_f1),

        "search_time": float(search_time),
        "training_time": float(training_time),

        "y_pred": y_pred,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print(
        "CAREERCAST MILESTONE 2 - "
        "SBERT EMBEDDING CLASSIFIER COMPARISON"
    )
    print("=" * 70)

    print()

    log(
        f"XGBoost version: {xgb.__version__}"
    )

    print(
        "Sentence-BERT embeddings: "
        "all-MiniLM-L6-v2"
    )

    print(
        "Embedding status: PRETRAINED "
        "(NOT fine-tuned)"
    )

    print()

    # ============================================================
    # STAGE 1
    # LOAD DATASET, EMBEDDINGS AND SAVED SPLIT
    # ============================================================

    log(
        "Stage 1: Loading dataset, embeddings, "
        "and saved train/test split..."
    )

    try:

        required_paths = [
            DATASET_PATH,
            EMBEDDINGS_PATH,
            SPLIT_PATH,
        ]

        for path in required_paths:

            if not os.path.isfile(path):
                fail(
                    f"Required file not found: '{path}'."
                )

        # --------------------------------------------------------
        # Load dataset
        # --------------------------------------------------------

        df = pd.read_csv(
            DATASET_PATH
        )

        required_columns = {
            "skills",
            "career",
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:

            fail(
                "Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # --------------------------------------------------------
        # Load embeddings
        # --------------------------------------------------------

        embeddings = np.load(
            EMBEDDINGS_PATH
        )

        # --------------------------------------------------------
        # Load EXACT split created by the SBERT script
        # --------------------------------------------------------

        split_data = np.load(
            SPLIT_PATH
        )

        train_idx = split_data["train_idx"]
        test_idx = split_data["test_idx"]

    except SystemExit:
        raise

    except Exception as e:

        fail(
            f"Failed to load required files: {e}",
            e,
        )

    # ============================================================
    # VALIDATE EMBEDDINGS
    # ============================================================

    if embeddings.ndim != 2:

        fail(
            "Embeddings array must be 2-dimensional. "
            f"Received shape: {embeddings.shape}"
        )

    if embeddings.shape[0] != len(df):

        fail(
            "Embeddings row count does not match dataset row count.\n"
            f"Dataset rows: {len(df)}\n"
            f"Embedding rows: {embeddings.shape[0]}\n\n"
            "Regenerate the Sentence-BERT embeddings using "
            "the same dataset."
        )

    # ============================================================
    # VALIDATE SPLIT
    # ============================================================

    if len(train_idx) == 0:

        fail(
            "Training split contains 0 samples."
        )

    if len(test_idx) == 0:

        fail(
            "Testing split contains 0 samples."
        )

    if (
        train_idx.min() < 0
        or test_idx.min() < 0
    ):

        fail(
            "Saved split contains negative indices."
        )

    if (
        train_idx.max() >= len(df)
        or test_idx.max() >= len(df)
    ):

        fail(
            "Saved split indices are outside "
            "the dataset range."
        )

    # ------------------------------------------------------------
    # Check overlap
    # ------------------------------------------------------------

    overlap = np.intersect1d(
        train_idx,
        test_idx,
    )

    if len(overlap) > 0:

        fail(
            "Train/test split contains overlapping indices. "
            "This would cause data leakage."
        )

    # ============================================================
    # DATASET INFORMATION
    # ============================================================

    log(
        f"Dataset rows: {len(df)}"
    )

    log(
        f"Embedding shape: {embeddings.shape}"
    )

    log(
        f"Training samples: {len(train_idx)}"
    )

    log(
        f"Testing samples: {len(test_idx)}"
    )

    log(
        "Exact saved train/test split reused."
    )

    # ============================================================
    # STAGE 2
    # ENCODE CAREER LABELS
    # ============================================================

    log(
        "Stage 2: Encoding career labels..."
    )

    try:

        careers = (
            df["career"]
            .astype(str)
            .values
        )

        skills_text = (
            df["skills"]
            .astype(str)
            .values
        )

        label_encoder = LabelEncoder()

        y_all = label_encoder.fit_transform(
            careers
        )

        # --------------------------------------------------------
        # Apply EXACT saved split
        # --------------------------------------------------------

        X_train = embeddings[
            train_idx
        ].astype(
            np.float32,
            copy=False,
        )

        X_test = embeddings[
            test_idx
        ].astype(
            np.float32,
            copy=False,
        )

        y_train = y_all[
            train_idx
        ]

        y_test = y_all[
            test_idx
        ]

        skills_test = skills_text[
            test_idx
        ]

    except Exception as e:

        fail(
            f"Failed to prepare training data: {e}",
            e,
        )

    # ============================================================
    # DATA INFORMATION
    # ============================================================

    n_careers = len(
        label_encoder.classes_
    )

    print()

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Embedding dimension: {X_train.shape[1]}"
    )

    print(
        f"Number of career classes: {n_careers}"
    )

    print()

    # ============================================================
    # STAGE 3
    # TRAIN THREE CLASSIFIERS
    # ============================================================

    print("=" * 70)
    print(
        "STAGE 3 - CLASSIFIER COMPARISON"
    )
    print("=" * 70)

    print()

    results = []

    # ============================================================
    # MODEL 1 - LOGISTIC REGRESSION
    # ============================================================

    logreg_base = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results.append(
        evaluate_model(
            "LogisticRegression",
            logreg_base,
            LOGREG_PARAM_GRID,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    print()

    # ============================================================
    # MODEL 2 - RANDOM FOREST
    # ============================================================

    rf_base = RandomForestClassifier(
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results.append(
        evaluate_model(
            "RandomForestClassifier",
            rf_base,
            RF_PARAM_GRID,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    print()

    # ============================================================
    # MODEL 3 - XGBOOST
    # ============================================================

    xgb_base = XGBClassifier(
        objective="multi:softprob",
        num_class=n_careers,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **XGB_FIXED_PARAMS,
    )

    results.append(
        evaluate_model(
            "XGBClassifier",
            xgb_base,
            XGB_PARAM_GRID,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    # ============================================================
    # STAGE 4
    # SELECT BEST MODEL USING CV ACCURACY
    # ============================================================

    print()

    print("=" * 70)
    print(
        "STAGE 4 - MODEL SELECTION"
    )
    print("=" * 70)

    print()

    log(
        "Selecting best model using "
        "CROSS-VALIDATION accuracy."
    )

    print(
        "IMPORTANT: Test accuracy is NOT used "
        "for model selection."
    )

    print()

    # ------------------------------------------------------------
    # IMPORTANT CORRECTION
    #
    # DO NOT use:
    #
    # max(results, key=lambda r: r["test_accuracy"])
    #
    # because that uses the test set to select the model.
    #
    # Instead use CV accuracy.
    # ------------------------------------------------------------

    best_result = max(
        results,
        key=lambda r: r["cv_accuracy"],
    )

    log(
        f"Selected model: {best_result['name']}"
    )

    print()

    print(
        f"Selected using CV accuracy: "
        f"{best_result['cv_accuracy']:.4f}"
    )

    print(
        f"Final test accuracy: "
        f"{best_result['test_accuracy']:.4f}"
    )

    print(
        f"Final macro precision: "
        f"{best_result['precision_macro']:.4f}"
    )

    print(
        f"Final macro recall: "
        f"{best_result['recall_macro']:.4f}"
    )

    print(
        f"Final macro F1: "
        f"{best_result['macro_f1']:.4f}"
    )

    # ============================================================
    # STAGE 5
    # PRINT COMPLETE MODEL COMPARISON
    # ============================================================

    print()

    print("=" * 70)
    print(
        "MODEL COMPARISON"
    )
    print("=" * 70)

    print()

    print(
        f"{'Model':<28}"
        f"{'CV Accuracy':<16}"
        f"{'Test Accuracy':<16}"
        f"{'Macro F1':<12}"
    )

    print("-" * 70)

    for result in results:

        marker = (
            " <-- SELECTED"
            if result["name"]
            == best_result["name"]
            else ""
        )

        print(
            f"{result['name']:<28}"
            f"{result['cv_accuracy']:<16.4f}"
            f"{result['test_accuracy']:<16.4f}"
            f"{result['macro_f1']:<12.4f}"
            f"{marker}"
        )

    # ============================================================
    # STAGE 6
    # CLASSIFICATION REPORT
    # ============================================================

    print()

    print("=" * 70)

    print(
        f"CLASSIFICATION REPORT - "
        f"{best_result['name']}"
    )

    print("=" * 70)

    best_pred = best_result[
        "y_pred"
    ]

    true_labels_test = (
        label_encoder.inverse_transform(
            y_test
        )
    )

    best_pred_labels = (
        label_encoder.inverse_transform(
            best_pred
        )
    )

    print()

    print(
        classification_report(
            true_labels_test,
            best_pred_labels,
            zero_division=0,
        )
    )

    # ============================================================
    # STAGE 7
    # SAVE ARTIFACTS
    # ============================================================

    print()

    print("=" * 70)

    print(
        "STAGE 7 - SAVING ARTIFACTS"
    )

    print("=" * 70)

    try:

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Label encoder
        # --------------------------------------------------------

        label_encoder_path = os.path.join(
            OUTPUT_DIR,
            "label_encoder.pkl",
        )

        joblib.dump(
            label_encoder,
            label_encoder_path,
        )

        # --------------------------------------------------------
        # Individual model paths
        # --------------------------------------------------------

        model_file_map = {

            "LogisticRegression":
                "logistic_regression_model.pkl",

            "RandomForestClassifier":
                "random_forest_model.pkl",

            "XGBClassifier":
                "xgboost_model.pkl",
        }

        saved_model_paths = {}

        # --------------------------------------------------------
        # Save all three models
        # --------------------------------------------------------

        for result in results:

            model_path = os.path.join(
                OUTPUT_DIR,
                model_file_map[
                    result["name"]
                ],
            )

            joblib.dump(
                result["model"],
                model_path,
            )

            saved_model_paths[
                result["name"]
            ] = model_path

        # --------------------------------------------------------
        # Save selected best model
        # --------------------------------------------------------

        best_model_path = os.path.join(
            OUTPUT_DIR,
            "best_model.pkl",
        )

        joblib.dump(
            best_result["model"],
            best_model_path,
        )

        # --------------------------------------------------------
        # Save metrics CSV
        # --------------------------------------------------------

        metrics_rows = []

        for result in results:

            metrics_rows.append({

                "model":
                    result["name"],

                "cv_accuracy":
                    result["cv_accuracy"],

                "test_accuracy":
                    result["test_accuracy"],

                "precision_macro":
                    result["precision_macro"],

                "recall_macro":
                    result["recall_macro"],

                "macro_f1":
                    result["macro_f1"],

                "search_time_seconds":
                    result["search_time"],

                "training_time_seconds":
                    result["training_time"],

                "best_params":
                    json.dumps(
                        result["best_params"]
                    ),

                "selected_by_cv":
                    result["name"]
                    == best_result["name"],
            })

        metrics_path = os.path.join(
            OUTPUT_DIR,
            "sbert_classifier_metrics.csv",
        )

        pd.DataFrame(
            metrics_rows
        ).to_csv(
            metrics_path,
            index=False,
        )

        # --------------------------------------------------------
        # Save predictions
        # --------------------------------------------------------

        predictions_path = os.path.join(
            OUTPUT_DIR,
            "sbert_classifier_predictions.csv",
        )

        predictions_df = pd.DataFrame({

            "skills":
                skills_test,

            "true_career":
                true_labels_test,

            "predicted_career":
                best_pred_labels,
        })

        predictions_df.to_csv(
            predictions_path,
            index=False,
        )

        # --------------------------------------------------------
        # Calculate total execution time
        # --------------------------------------------------------

        total_time = (
            time.time()
            - SCRIPT_START
        )

        # --------------------------------------------------------
        # Save summary JSON
        # --------------------------------------------------------

        summary = {

            "project":
                "CareerCast",

            "milestone":
                "Milestone 2",

            "pipeline":
                "Sentence-BERT embeddings + classifier comparison",

            "embedding_model":
                "all-MiniLM-L6-v2",

            "embedding_fine_tuned":
                False,

            "embedding_note":
                "Pretrained Sentence-BERT embeddings were used. "
                "No SBERT fine-tuning was performed.",

            "embedding_dimension":
                int(X_train.shape[1]),

            "dataset_rows":
                int(len(df)),

            "training_samples":
                int(len(X_train)),

            "testing_samples":
                int(len(X_test)),

            "number_of_career_classes":
                int(n_careers),

            "random_state":
                RANDOM_STATE,

            "cv_folds":
                CV_FOLDS,

            "model_selection_metric":
                "cross_validation_accuracy",

            "test_set_used_for_model_selection":
                False,

            "best_model":
                best_result["name"],

            "best_model_cv_accuracy":
                best_result["cv_accuracy"],

            "best_model_test_accuracy":
                best_result["test_accuracy"],

            "best_model_precision_macro":
                best_result["precision_macro"],

            "best_model_recall_macro":
                best_result["recall_macro"],

            "best_model_macro_f1":
                best_result["macro_f1"],

            "best_model_params":
                best_result["best_params"],

            "split_reused_from":
                SPLIT_PATH,

            "embedding_source":
                EMBEDDINGS_PATH,

            "total_time_seconds":
                total_time,

            "all_models": [

                {
                    "model":
                        result["name"],

                    "cv_accuracy":
                        result["cv_accuracy"],

                    "test_accuracy":
                        result["test_accuracy"],

                    "precision_macro":
                        result["precision_macro"],

                    "recall_macro":
                        result["recall_macro"],

                    "macro_f1":
                        result["macro_f1"],

                    "best_params":
                        result["best_params"],

                    "selected":
                        result["name"]
                        == best_result["name"],
                }

                for result in results
            ],
        }

        summary_path = os.path.join(
            OUTPUT_DIR,
            "sbert_classifier_summary.json",
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
            )

    except Exception as e:

        fail(
            f"Failed while saving artifacts: {e}",
            e,
        )

    # ============================================================
    # STAGE 8
    # VERIFY FILES
    # ============================================================

    print()

    log(
        "Stage 8: Verifying saved files..."
    )

    expected_files = [

        label_encoder_path,

        best_model_path,

        metrics_path,

        predictions_path,

        summary_path,

    ] + list(
        saved_model_paths.values()
    )

    missing_after_save = [

        path
        for path in expected_files
        if not os.path.isfile(path)

    ]

    if missing_after_save:

        fail(
            "The following expected files were "
            "not found after saving:\n"
            + "\n".join(
                missing_after_save
            )
        )

    log(
        "All output files verified successfully."
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    total_time = (
        time.time()
        - SCRIPT_START
    )

    print()

    print("=" * 70)

    print(
        "SBERT EMBEDDING CLASSIFIER "
        "COMPARISON COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "Sentence-BERT model:"
    )

    print(
        "  all-MiniLM-L6-v2 "
        "(pretrained, not fine-tuned)"
    )

    print()

    print(
        "Classifier results:"
    )

    print()

    for result in results:

        marker = (
            " <-- SELECTED BEST MODEL"
            if result["name"]
            == best_result["name"]
            else ""
        )

        print(
            f"  {result['name']}"
        )

        print(
            f"      CV accuracy   : "
            f"{result['cv_accuracy']:.4f}"
        )

        print(
            f"      Test accuracy : "
            f"{result['test_accuracy']:.4f}"
        )

        print(
            f"      Macro F1      : "
            f"{result['macro_f1']:.4f}"
        )

        print(
            f"      Best params   : "
            f"{result['best_params']}"
        )

        print(
            f"      {marker}"
        )

        print()

    print(
        "FINAL SELECTED MODEL:"
    )

    print(
        f"  {best_result['name']}"
    )

    print()

    print(
        f"  Selection CV accuracy : "
        f"{best_result['cv_accuracy']:.4f}"
    )

    print(
        f"  Final test accuracy   : "
        f"{best_result['test_accuracy']:.4f}"
    )

    print(
        f"  Macro precision       : "
        f"{best_result['precision_macro']:.4f}"
    )

    print(
        f"  Macro recall          : "
        f"{best_result['recall_macro']:.4f}"
    )

    print(
        f"  Macro F1              : "
        f"{best_result['macro_f1']:.4f}"
    )

    print()

    print(
        f"Total execution time: "
        f"{total_time:.1f} seconds"
    )

    print()

    print(
        "Saved files:"
    )

    for path in expected_files:

        print(
            f"  {path}"
        )

    print()

    print("=" * 70)

    print(
        "IMPORTANT:"
    )

    print(
        "The test set was kept separate from model "
        "selection. The best classifier was selected "
        "using cross-validation accuracy only."
    )

    print("=" * 70)


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":

    try:

        main()

    except SystemExit:

        raise

    except Exception as e:

        fail(
            "Unhandled error in main().",
            e,
        )
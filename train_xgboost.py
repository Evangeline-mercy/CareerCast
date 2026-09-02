"""
CareerCast Milestone 2 — XGBoost Training Script
=================================================
Safe, fast, fully logged XGBoost with cross-validated hyperparameter tuning.
878-class multiclass problem using pre-built tree features (200 SVD + 6 RIASEC).

NOTE: Job_Description is intentionally excluded because each career has exactly
one unique job description, making it a direct target leakage feature.

Run from project root:
    venv\\Scripts\\activate
    python train_xgboost.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
import json
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts():
    """Return a compact timestamp string for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    """Print a timestamped log line and flush immediately."""
    print(f"[{ts()}] {msg}", flush=True)


def elapsed(start):
    """Return seconds elapsed since start."""
    return time.time() - start


def top_k_accuracy(y_true, proba, k):
    """
    Compute Top-K accuracy.
    proba: (n_samples, n_classes) probability matrix
    """
    top_k_preds = np.argsort(proba, axis=1)[:, -k:]
    correct = sum(
        y_true[i] in top_k_preds[i] for i in range(len(y_true))
    )
    return correct / len(y_true)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURE_DIR = os.path.join("results", "tree_features")
OUTPUT_DIR  = os.path.join("results", "xgboost_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_start = time.time()
    log("=" * 65)
    log("CareerCast Milestone 2 — XGBoost Training")
    log("=" * 65)

    # -----------------------------------------------------------------------
    # Stage 1: Load pre-built features
    # -----------------------------------------------------------------------
    log("Stage 1/6  Loading pre-built tree features …")
    try:
        stage_start = time.time()

        X_train = np.load(os.path.join(FEATURE_DIR, "X_train.npy")).astype(np.float32)
        X_test  = np.load(os.path.join(FEATURE_DIR, "X_test.npy")).astype(np.float32)
        y_train = np.load(os.path.join(FEATURE_DIR, "y_train.npy"))
        y_test  = np.load(os.path.join(FEATURE_DIR, "y_test.npy"))
        label_encoder = joblib.load(os.path.join(FEATURE_DIR, "label_encoder.joblib"))

        log(f"  Loaded in {elapsed(stage_start):.1f}s")
    except Exception:
        log("ERROR loading features:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 2: Verify and print startup info
    # -----------------------------------------------------------------------
    log("Stage 2/6  Verifying data …")
    try:
        n_classes  = len(label_encoder.classes_)
        n_features = X_train.shape[1]
        n_train    = X_train.shape[0]
        n_test     = X_test.shape[0]

        N_ITER = 8
        N_FOLDS = 3
        TOTAL_FITS = N_ITER * N_FOLDS

        print(flush=True)
        print("  ── Pre-training summary ──────────────────────────", flush=True)
        print(f"  X_train shape       : {X_train.shape}", flush=True)
        print(f"  X_test shape        : {X_test.shape}", flush=True)
        print(f"  X_train dtype       : {X_train.dtype}", flush=True)
        print(f"  Number of classes   : {n_classes}", flush=True)
        print(f"  Number of features  : {n_features}", flush=True)
        print(f"  XGBoost version     : {xgb.__version__}", flush=True)
        print(f"  CV candidates       : {N_ITER}", flush=True)
        print(f"  CV folds            : {N_FOLDS}", flush=True)
        print(f"  Expected total fits : {TOTAL_FITS}  (+ 1 final refit)", flush=True)
        print(f"  n_estimators range  : [20, 30, 40, 50]", flush=True)
        print(f"  max_depth range     : [2, 3, 4]", flush=True)
        print(f"  WARNING: 878-class XGBoost trains up to", flush=True)
        print(f"           50 × 878 = 43,900 trees per fit.", flush=True)
        print(f"           Be patient — progress is logged.", flush=True)
        print("  ─────────────────────────────────────────────────", flush=True)
        print(flush=True)

        # Sanity checks
        assert X_train.shape[0] == y_train.shape[0], "X_train / y_train row mismatch"
        assert X_test.shape[0]  == y_test.shape[0],  "X_test / y_test row mismatch"
        assert X_train.shape[1] == X_test.shape[1],  "Feature dimension mismatch"
        log("  Data verified OK.")
    except Exception:
        log("ERROR verifying data:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 3: Define parameter space and CV strategy
    # -----------------------------------------------------------------------
    log("Stage 3/6  Setting up hyperparameter search …")
    try:
        param_dist = {
    "n_estimators": [10, 15, 20],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1, 0.15],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

        base_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",       
            n_jobs=1,            
            random_state=42,
            verbosity=0,           
        
    
        )

        cv_strategy = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True, 
            random_state=42
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=N_ITER,
            cv=cv_strategy,
            scoring="accuracy",
            n_jobs=1,          
            refit=True,
            random_state=42,
            verbose=2,          
            error_score="raise",
        )

        log(f"  RandomizedSearchCV configured: {N_ITER} iter × {N_FOLDS} folds = {TOTAL_FITS} fits")
    except Exception:
        log("ERROR setting up search:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 4: Run CV search
    # -----------------------------------------------------------------------
    log("Stage 4/6  Running RandomizedSearchCV (this is the slow part) …")
    log("           Progress lines will appear below from sklearn verbose=2.")
    log("           Each 'fit N' line confirms the script is NOT frozen.")
    try:
        cv_start = time.time()
        search.fit(X_train, y_train)
        cv_time = elapsed(cv_start)

        log(f"  RandomizedSearchCV finished in {cv_time:.1f}s  ({cv_time/60:.1f} min)")
        log(f"  Best CV accuracy : {search.best_score_:.4f}")
        log(f"  Best parameters  : {search.best_params_}")
    except Exception:
        log("ERROR during RandomizedSearchCV:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 5: Evaluate on held-out test set
    # -----------------------------------------------------------------------
    log("Stage 5/6  Evaluating best model on test set …")
    try:
        eval_start = time.time()

        best_model = search.best_estimator_

        # predict_proba returns (n_test, n_classes) — ~3512 × 878 ≈ 24 MB float32
        log("    Computing predict_proba …")
        proba = best_model.predict_proba(X_test).astype(np.float32)
        log(f"    proba shape: {proba.shape}  dtype: {proba.dtype}")

        # Top-K accuracies
        top1 = top_k_accuracy(y_test, proba, 1)
        top3 = top_k_accuracy(y_test, proba, 3)
        top5 = top_k_accuracy(y_test, proba, 5)
        log(f"    Top-1 Accuracy : {top1:.4f}")
        log(f"    Top-3 Accuracy : {top3:.4f}")
        log(f"    Top-5 Accuracy : {top5:.4f}")

        # Hard predictions for sklearn metrics
        y_pred = np.argmax(proba, axis=1)

        # Free the large probability matrix
        del proba

        macro_precision  = precision_score(y_test, y_pred, average="macro",    zero_division=0)
        macro_recall     = recall_score(   y_test, y_pred, average="macro",    zero_division=0)
        macro_f1         = f1_score(       y_test, y_pred, average="macro",    zero_division=0)
        weighted_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        weighted_recall    = recall_score(   y_test, y_pred, average="weighted", zero_division=0)
        weighted_f1        = f1_score(       y_test, y_pred, average="weighted", zero_division=0)

        eval_time = elapsed(eval_start)
        total_time = elapsed(script_start)

        print(flush=True)
        print("  ── Test-set results ──────────────────────────────", flush=True)
        print(f"  Top-1 Accuracy     : {top1:.4f}", flush=True)
        print(f"  Top-3 Accuracy     : {top3:.4f}", flush=True)
        print(f"  Top-5 Accuracy     : {top5:.4f}", flush=True)
        print(f"  Macro Precision    : {macro_precision:.4f}", flush=True)
        print(f"  Macro Recall       : {macro_recall:.4f}", flush=True)
        print(f"  Macro F1           : {macro_f1:.4f}", flush=True)
        print(f"  Weighted Precision : {weighted_precision:.4f}", flush=True)
        print(f"  Weighted Recall    : {weighted_recall:.4f}", flush=True)
        print(f"  Weighted F1        : {weighted_f1:.4f}", flush=True)
        print("  ─────────────────────────────────────────────────", flush=True)
        print(flush=True)

        # Full classification report (string)
        report_str = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0,
            output_dict=False,
        )
        report_dict = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0,
            output_dict=True,
        )

    except Exception:
        log("ERROR during evaluation:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 6: Save artifacts
    # -----------------------------------------------------------------------
    log("Stage 6/6  Saving artifacts …")
    try:
        # 6a. Model
        model_path = os.path.join(OUTPUT_DIR, "xgboost_model.joblib")
        log(f"    Saving model → {model_path}")
        joblib.dump(best_model, model_path, compress=3)

        # 6b. Label encoder (copy, not re-fit)
        le_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
        log(f"    Saving label encoder → {le_path}")
        joblib.dump(label_encoder, le_path)

        # 6c. CV results CSV
        import pandas as pd
        cv_results_path = os.path.join(OUTPUT_DIR, "cv_results.csv")
        log(f"    Saving CV results → {cv_results_path}")
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(cv_results_path, index=False)

        # 6d. Classification report JSON
        report_path = os.path.join(OUTPUT_DIR, "classification_report.json")
        log(f"    Saving classification report → {report_path}")
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        # 6e. metadata.json
        metadata = {
            "xgboost_version"      : xgb.__version__,
            "num_classes"          : n_classes,
            "training_samples"     : int(n_train),
            "test_samples"         : int(n_test),
            "num_features"         : int(n_features),
            "feature_description"  : "200 SVD components + 6 RIASEC features (float32)",
            "leakage_note"         : (
                "Job_Description was intentionally excluded because every career has "
                "exactly one unique job description (mean=1, min=1, max=1 per career), "
                "making it a direct target leakage feature."
            ),
            "cv_strategy"          : f"RandomizedSearchCV with StratifiedKFold(n_splits={N_FOLDS})",
            "n_search_iterations"  : N_ITER,
            "total_cv_fits"        : TOTAL_FITS,
            "best_cv_accuracy"     : float(round(search.best_score_, 6)),
            "best_parameters"      : search.best_params_,
            "fixed_parameters"     : {
                "objective"   : "multi:softprob",
                "tree_method" : "hist",
                "n_jobs"      : 1,
                "random_state": 42,
                "verbosity"   : 0,
            },
            "top1_accuracy"        : float(round(top1, 6)),
            "top3_accuracy"        : float(round(top3, 6)),
            "top5_accuracy"        : float(round(top5, 6)),
            "macro_precision"      : float(round(macro_precision,   6)),
            "macro_recall"         : float(round(macro_recall,      6)),
            "macro_f1"             : float(round(macro_f1,          6)),
            "weighted_precision"   : float(round(weighted_precision, 6)),
            "weighted_recall"      : float(round(weighted_recall,    6)),
            "weighted_f1"          : float(round(weighted_f1,        6)),
            "training_time_seconds": float(round(cv_time,   2)),
            "evaluation_time_seconds": float(round(eval_time, 2)),
            "total_time_seconds"   : float(round(total_time, 2)),
            "random_state"         : 42,
        }

        meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
        log(f"    Saving metadata → {meta_path}")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log("  All artifacts saved successfully.")
    except Exception:
        log("ERROR saving artifacts:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    log("=" * 65)
    log(f"XGBoost training complete!  Total time: {total_time:.1f}s  ({total_time/60:.1f} min)")
    log(f"Artifacts written to: {OUTPUT_DIR}/")
    log("=" * 65)


if __name__ == "__main__":
    main()
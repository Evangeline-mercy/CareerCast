"""
CareerCast - Milestone 2
MODEL TUNING VALIDATION

Checks the EXISTING Random Forest and XGBoost tuning artifacts.

This script:
- Does NOT modify the V4 recommender
- Does NOT modify V4 demo careers
- Does NOT modify existing trained models
- Does NOT modify the dataset
- Verifies cross-validation and hyperparameter tuning
- Compares Random Forest and XGBoost
"""

import os
import json
import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RF_DIR = "results/tree_model"
XGB_DIR = "results/xgboost_model"

RF_MODEL = os.path.join(
    RF_DIR,
    "random_forest_model.joblib"
)

RF_METADATA = os.path.join(
    RF_DIR,
    "metadata.json"
)

XGB_MODEL = os.path.join(
    XGB_DIR,
    "xgboost_model.joblib"
)

XGB_METADATA = os.path.join(
    XGB_DIR,
    "metadata.json"
)

XGB_LABEL_ENCODER = os.path.join(
    XGB_DIR,
    "label_encoder.joblib"
)

MODEL_COMPARISON = "model_comparison.csv"


# ============================================================
# HELPERS
# ============================================================

def separator():
    print("\n" + "=" * 70)


def check_file(path, name):
    if os.path.exists(path):
        print(f"[FOUND] {name}")
        print(f"        {path}")
        return True

    print(f"[MISSING] {name}")
    print(f"          {path}")
    return False


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MAIN
# ============================================================

def main():

    separator()

    print("CAREERCAST MILESTONE 2")
    print("MODEL TUNING VALIDATION")
    print("=" * 70)

    print("\nREAD-ONLY VALIDATION")
    print("Existing models will NOT be retrained.")
    print("Existing models will NOT be modified.")
    print("V4 recommender will NOT be modified.")
    print("V4 demo careers will NOT be modified.")
    print("Dataset will NOT be modified.")

    # ========================================================
    # 1. CHECK RANDOM FOREST ARTIFACTS
    # ========================================================

    separator()
    print("1. RANDOM FOREST ARTIFACTS")
    separator()

    rf_model_found = check_file(
        RF_MODEL,
        "Random Forest model"
    )

    rf_metadata_found = check_file(
        RF_METADATA,
        "Random Forest metadata"
    )

    # ========================================================
    # 2. CHECK XGBOOST ARTIFACTS
    # ========================================================

    separator()
    print("2. XGBOOST ARTIFACTS")
    separator()

    xgb_model_found = check_file(
        XGB_MODEL,
        "XGBoost model"
    )

    xgb_metadata_found = check_file(
        XGB_METADATA,
        "XGBoost metadata"
    )

    xgb_encoder_found = check_file(
        XGB_LABEL_ENCODER,
        "XGBoost label encoder"
    )

    # ========================================================
    # 3. RANDOM FOREST VALIDATION
    # ========================================================

    separator()
    print("3. RANDOM FOREST CROSS-VALIDATION")
    separator()

    rf_ok = False

    if rf_metadata_found:

        rf_meta = load_json(
            RF_METADATA
        )

        print(
            "Model:",
            rf_meta.get(
                "model_name",
                "RandomForestClassifier"
            )
        )

        print(
            "CV Strategy:",
            rf_meta.get(
                "cv_strategy",
                "Not available"
            )
        )

        print(
            "Search iterations:",
            rf_meta.get(
                "search_n_iter",
                "Not available"
            )
        )

        print(
            "Best CV Accuracy:",
            rf_meta.get(
                "best_cv_accuracy",
                "Not available"
            )
        )

        print(
            "Top-1 Accuracy:",
            rf_meta.get(
                "top1_accuracy",
                "Not available"
            )
        )

        print(
            "Top-3 Accuracy:",
            rf_meta.get(
                "top3_accuracy",
                "Not available"
            )
        )

        print(
            "Top-5 Accuracy:",
            rf_meta.get(
                "top5_accuracy",
                "Not available"
            )
        )

        print(
            "Macro Precision:",
            rf_meta.get(
                "macro_precision",
                "Not available"
            )
        )

        print(
            "Macro Recall:",
            rf_meta.get(
                "macro_recall",
                "Not available"
            )
        )

        print(
            "Macro F1:",
            rf_meta.get(
                "macro_f1",
                "Not available"
            )
        )

        print(
            "Best Parameters:"
        )

        print(
            rf_meta.get(
                "best_params",
                "Not available"
            )
        )

        cv_strategy = str(
            rf_meta.get(
                "cv_strategy",
                ""
            )
        )

        if (
            "StratifiedKFold" in cv_strategy
            and rf_model_found
        ):
            rf_ok = True

            print(
                "\nRandom Forest tuning check: PASS"
            )

        else:
            print(
                "\nRandom Forest tuning check: REVIEW"
            )

    # ========================================================
    # 4. XGBOOST VALIDATION
    # ========================================================

    separator()
    print("4. XGBOOST CROSS-VALIDATION")
    separator()

    xgb_ok = False

    if xgb_metadata_found:

        xgb_meta = load_json(
            XGB_METADATA
        )

        print(
            "XGBoost version:",
            xgb_meta.get(
                "xgboost_version",
                "Not available"
            )
        )

        print(
            "Number of classes:",
            xgb_meta.get(
                "num_classes",
                "Not available"
            )
        )

        print(
            "Number of features:",
            xgb_meta.get(
                "num_features",
                "Not available"
            )
        )

        print(
            "CV Strategy:",
            xgb_meta.get(
                "cv_strategy",
                "Not available"
            )
        )

        print(
            "Search iterations:",
            xgb_meta.get(
                "n_search_iterations",
                "Not available"
            )
        )

        print(
            "Total CV fits:",
            xgb_meta.get(
                "total_cv_fits",
                "Not available"
            )
        )

        print(
            "Best CV Accuracy:",
            xgb_meta.get(
                "best_cv_accuracy",
                "Not available"
            )
        )

        print(
            "Top-1 Accuracy:",
            xgb_meta.get(
                "top1_accuracy",
                "Not available"
            )
        )

        print(
            "Top-3 Accuracy:",
            xgb_meta.get(
                "top3_accuracy",
                "Not available"
            )
        )

        print(
            "Top-5 Accuracy:",
            xgb_meta.get(
                "top5_accuracy",
                "Not available"
            )
        )

        print(
            "Macro Precision:",
            xgb_meta.get(
                "macro_precision",
                "Not available"
            )
        )

        print(
            "Macro Recall:",
            xgb_meta.get(
                "macro_recall",
                "Not available"
            )
        )

        print(
            "Macro F1:",
            xgb_meta.get(
                "macro_f1",
                "Not available"
            )
        )

        print(
            "\nBest Parameters:"
        )

        print(
            xgb_meta.get(
                "best_parameters",
                "Not available"
            )
        )

        cv_strategy = str(
            xgb_meta.get(
                "cv_strategy",
                ""
            )
        )

        if (
            "RandomizedSearchCV" in cv_strategy
            and "StratifiedKFold" in cv_strategy
            and xgb_model_found
            and xgb_encoder_found
        ):
            xgb_ok = True

            print(
                "\nXGBoost tuning check: PASS"
            )

        else:
            print(
                "\nXGBoost tuning check: REVIEW"
            )

    # ========================================================
    # 5. MODEL COMPARISON
    # ========================================================

    separator()
    print("5. MODEL COMPARISON")
    separator()

    if os.path.exists(
        MODEL_COMPARISON
    ):

        comparison = pd.read_csv(
            MODEL_COMPARISON
        )

        print(
            comparison.to_string(
                index=False
            )
        )

        print(
            "\nModel comparison file: PASS"
        )

    else:

        print(
            "[INFO] model_comparison.csv "
            "not found in project root."
        )

        print(
            "Existing RF/XGBoost metadata "
            "will still be validated."
        )

    # ========================================================
    # 6. LOAD MODELS - READ ONLY
    # ========================================================

    separator()
    print("6. MODEL LOAD CHECK")
    separator()

    if rf_model_found:

        try:

            rf_model = joblib.load(
                RF_MODEL
            )

            print(
                "Random Forest model loading: PASS"
            )

            if hasattr(
                rf_model,
                "n_features_in_"
            ):

                print(
                    "RF input features:",
                    rf_model.n_features_in_
                )

        except Exception as e:

            print(
                "Random Forest model loading: FAILED"
            )

            print(
                e
            )

    if xgb_model_found:

        try:

            xgb_model = joblib.load(
                XGB_MODEL
            )

            print(
                "XGBoost model loading: PASS"
            )

            if hasattr(
                xgb_model,
                "n_features_in_"
            ):

                print(
                    "XGBoost input features:",
                    xgb_model.n_features_in_
                )

            if hasattr(
                xgb_model,
                "n_classes_"
            ):

                print(
                    "XGBoost classes:",
                    xgb_model.n_classes_
                )

        except Exception as e:

            print(
                "XGBoost model loading: FAILED"
            )

            print(
                e
            )

    # ========================================================
    # 7. FINAL STATUS
    # ========================================================

    separator()
    print("FINAL MILESTONE 2 MODEL TUNING STATUS")
    separator()

    if rf_ok:
        print(
            "Random Forest + CV tuning : PASS"
        )
    else:
        print(
            "Random Forest + CV tuning : REVIEW"
        )

    if xgb_ok:
        print(
            "XGBoost + CV tuning        : PASS"
        )
    else:
        print(
            "XGBoost + CV tuning        : REVIEW"
        )

    print()

    if rf_ok and xgb_ok:

        print(
            "MILESTONE 2 MODEL TUNING: PASS"
        )

        print(
            "\nRequirement satisfied:"
        )

        print(
            "Random Forest and XGBoost"
        )

        print(
            "with cross-validated"
        )

        print(
            "hyperparameter tuning."
        )

    else:

        print(
            "MILESTONE 2 MODEL TUNING: REVIEW"
        )

    separator()

    print(
        "\nIMPORTANT:"
    )

    print(
        "Your working V4 presentation demo "
        "was NOT changed."
    )

    print(
        "Python Engineer"
    )

    print(
        "AI/ML Engineer"
    )

    print(
        "Data Scientist"
    )

    print(
        "Embedded Systems Engineer"
    )

    print(
        "IoT Engineer"
    )

    separator()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
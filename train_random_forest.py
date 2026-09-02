# ============================================================
# CAREERCAST MILESTONE 2
# RANDOM FOREST CLASSIFIER
#
# Features:
# - TF-IDF text vectorization
# - Random Forest classifier
# - 5-fold cross-validation
# - GridSearchCV hyperparameter tuning
# - Train/Test evaluation
# - Model + vectorizer + label encoder saving
#
# Existing Milestone 1 files are NOT modified.
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    "results/careercast_milestone2_dataset.csv"
)

OUTPUT_ROOT = (
    "results/random_forest_model"
)

MODEL_PATH = os.path.join(
    OUTPUT_ROOT,
    "random_forest_model.joblib"
)

VECTORIZER_PATH = os.path.join(
    OUTPUT_ROOT,
    "tfidf_vectorizer.joblib"
)

LABEL_ENCODER_PATH = os.path.join(
    OUTPUT_ROOT,
    "label_encoder.joblib"
)

METRICS_PATH = os.path.join(
    OUTPUT_ROOT,
    "metrics.json"
)


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# TEST SIZE
# ============================================================

TEST_SIZE = 0.20


# ============================================================
# TOP-K
# ============================================================

TOP_K = 5


# ============================================================
# STEP 1
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 75)

    print(
        "1. LOADING DATASET"
    )

    print("=" * 75)

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        "Dataset shape:",
        df.shape
    )

    print(
        "Columns:"
    )

    print(
        list(df.columns)
    )

    required_columns = [

        "Career",
        "Essential_Skills",
        "Software_Skills",
        "Education_Level",
        "Experience_Level",
        "Job_Description"
    ]

    missing_columns = [

        column

        for column
        in required_columns

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    print(
        "\nRequired columns: PASS"
    )

    print(
        "Unique careers:",
        df["Career"].nunique()
    )

    return df


# ============================================================
# STEP 2
# CLEAN TEXT
# ============================================================

def clean_text(value):

    if pd.isna(value):

        return ""

    return str(value).strip()


# ============================================================
# STEP 3
# BUILD COMBINED TEXT
# ============================================================

def build_text_features(df):

    print("\n" + "=" * 75)

    print(
        "2. BUILDING TEXT FEATURES"
    )

    print("=" * 75)

    df = df.copy()

    df["combined_text"] = (

        df["Essential_Skills"]
        .apply(clean_text)

        + " "

        +

        df["Software_Skills"]
        .apply(clean_text)

        + " "

        +

        df["Education_Level"]
        .apply(clean_text)

        + " "

        +

        df["Experience_Level"]
        .apply(clean_text)

        + " "

        +

        df["Job_Description"]
        .apply(clean_text)
    )

    df["combined_text"] = (
        df["combined_text"]
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    print(
        "Combined text created: PASS"
    )

    print(
        "\nExample:"
    )

    print(
        df["combined_text"].iloc[0][:500]
    )

    return df


# ============================================================
# STEP 4
# REMOVE INVALID ROWS
# ============================================================

def clean_dataset(df):

    print("\n" + "=" * 75)

    print(
        "3. CLEANING DATASET"
    )

    print("=" * 75)

    before = len(df)

    df = df[
        df["Career"].notna()
    ]

    df = df[
        df["combined_text"].str.len() > 0
    ]

    df = df.reset_index(
        drop=True
    )

    after = len(df)

    print(
        "Rows before cleaning:",
        before
    )

    print(
        "Rows after cleaning:",
        after
    )

    print(
        "Rows removed:",
        before - after
    )

    return df


# ============================================================
# STEP 5
# PREPARE X AND Y
# ============================================================

def prepare_labels(df):

    print("\n" + "=" * 75)

    print(
        "4. PREPARING LABELS"
    )

    print("=" * 75)

    X_text = (
        df["combined_text"]
        .values
    )

    y_text = (
        df["Career"]
        .astype(str)
        .values
    )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_text
    )

    print(
        "Number of careers:",
        len(label_encoder.classes_)
    )

    print(
        "Label encoding: PASS"
    )

    return (
        X_text,
        y,
        label_encoder
    )


# ============================================================
# STEP 6
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(
    X_text,
    y
):

    print("\n" + "=" * 75)

    print(
        "5. TRAIN / TEST SPLIT"
    )

    print("=" * 75)

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X_text,
            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y
        )
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    print(
        "Test size:",
        TEST_SIZE
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# STEP 7
# TF-IDF
# ============================================================

def build_tfidf(
    X_train,
    X_test
):

    print("\n" + "=" * 75)

    print(
        "6. TF-IDF VECTORIZATION"
    )

    print("=" * 75)

    vectorizer = TfidfVectorizer(

        lowercase=True,

        stop_words="english",

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True
    )

    X_train_tfidf = (
        vectorizer.fit_transform(
            X_train
        )
    )

    X_test_tfidf = (
        vectorizer.transform(
            X_test
        )
    )

    print(
        "Training TF-IDF shape:",
        X_train_tfidf.shape
    )

    print(
        "Testing TF-IDF shape:",
        X_test_tfidf.shape
    )

    print(
        "TF-IDF vocabulary:",
        len(
            vectorizer.vocabulary_
        )
    )

    return (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf
    )


# ============================================================
# STEP 8
# RANDOM FOREST
# CROSS-VALIDATED HYPERPARAMETER TUNING
# ============================================================

def tune_random_forest(
    X_train,
    y_train
):

    print("\n" + "=" * 75)

    print(
        "7. RANDOM FOREST HYPERPARAMETER TUNING"
    )

    print("=" * 75)

    random_forest = RandomForestClassifier(

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    # --------------------------------------------------------
    # Hyperparameter grid
    # --------------------------------------------------------

    parameter_grid = {

        "n_estimators": [
            100,
            200
        ],

        "max_depth": [
            None,
            20,
            40
        ],

        "min_samples_split": [
            2,
            5
        ],

        "min_samples_leaf": [
            1,
            2
        ],

        "max_features": [
            "sqrt"
        ]
    }

    # --------------------------------------------------------
    # 5-Fold Stratified Cross Validation
    # --------------------------------------------------------

    cross_validator = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    print(
        "Cross-validation folds:",
        5
    )

    print(
        "Hyperparameter tuning: STARTED"
    )

    # --------------------------------------------------------
    # Grid Search
    # --------------------------------------------------------

    grid_search = GridSearchCV(

        estimator=random_forest,

        param_grid=parameter_grid,

        scoring="accuracy",

        cv=cross_validator,

        n_jobs=-1,

        verbose=2,

        refit=True
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print(
        "\nHyperparameter tuning: COMPLETE"
    )

    print(
        "\nBest parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest CV accuracy:",
        round(
            grid_search.best_score_,
            6
        )
    )

    return grid_search


# ============================================================
# STEP 9
# TEST EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    label_encoder
):

    print("\n" + "=" * 75)

    print(
        "8. RANDOM FOREST TEST EVALUATION"
    )

    print("=" * 75)

    predictions = (
        model.predict(
            X_test
        )
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_test,

        predictions
    )

    print(
        "Test Accuracy:",
        round(
            accuracy,
            6
        )
    )

    # --------------------------------------------------------
    # Top-K accuracy
    # --------------------------------------------------------

    top_k_accuracy = (
        top_k_accuracy_score(

            y_test,

            probabilities,

            k=TOP_K,

            labels=np.arange(
                len(
                    label_encoder.classes_
                )
            )
        )
    )

    print(
        f"Top-{TOP_K} Accuracy:",
        round(
            top_k_accuracy,
            6
        )
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(

        y_test,

        predictions,

        labels=np.arange(
            len(
                label_encoder.classes_
            )
        ),

        target_names=(
            label_encoder.classes_
        ),

        zero_division=0,

        output_dict=True
    )

    print(
        "\nClassification report generated."
    )

    return (
        accuracy,
        top_k_accuracy,
        report
    )


# ============================================================
# STEP 10
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    model,
    vectorizer,
    label_encoder,
    accuracy,
    top_k_accuracy,
    report,
    best_params,
    best_cv_score
):

    print("\n" + "=" * 75)

    print(
        "9. SAVING RANDOM FOREST ARTIFACTS"
    )

    print("=" * 75)

    os.makedirs(

        OUTPUT_ROOT,

        exist_ok=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    joblib.dump(

        model,

        MODEL_PATH
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    joblib.dump(

        vectorizer,

        VECTORIZER_PATH
    )

    # --------------------------------------------------------
    # Label encoder
    # --------------------------------------------------------

    joblib.dump(

        label_encoder,

        LABEL_ENCODER_PATH
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {

        "model":
            "Random Forest",

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "cross_validation_folds":
            5,

        "best_cv_accuracy":
            float(
                best_cv_score
            ),

        "test_accuracy":
            float(
                accuracy
            ),

        "top_k":
            TOP_K,

        "top_k_accuracy":
            float(
                top_k_accuracy
            ),

        "best_parameters":
            best_params,

        "classification_report":
            report
    }

    with open(

        METRICS_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=2
        )

    print(
        "Model saved:",
        MODEL_PATH
    )

    print(
        "Vectorizer saved:",
        VECTORIZER_PATH
    )

    print(
        "Label encoder saved:",
        LABEL_ENCODER_PATH
    )

    print(
        "Metrics saved:",
        METRICS_PATH
    )


# ============================================================
# STEP 11
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 75)

    print(
        "CAREERCAST MILESTONE 2"
    )

    print(
        "RANDOM FOREST CLASSIFIER"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Build combined text
    # --------------------------------------------------------

    df = build_text_features(
        df
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_dataset(
        df
    )

    # --------------------------------------------------------
    # Prepare labels
    # --------------------------------------------------------

    (
        X_text,
        y,
        label_encoder
    ) = prepare_labels(
        df
    )

    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(

        X_text,

        y
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    (
        vectorizer,
        X_train_tfidf,
        X_test_tfidf
    ) = build_tfidf(

        X_train,

        X_test
    )

    # --------------------------------------------------------
    # Random Forest tuning
    # --------------------------------------------------------

    grid_search = (
        tune_random_forest(

            X_train_tfidf,

            y_train
        )
    )

    best_model = (
        grid_search.best_estimator_
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    (
        accuracy,
        top_k_accuracy,
        report
    ) = evaluate_model(

        best_model,

        X_test_tfidf,

        y_test,

        label_encoder
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_artifacts(

        best_model,

        vectorizer,

        label_encoder,

        accuracy,

        top_k_accuracy,

        report,

        grid_search.best_params_,

        grid_search.best_score_
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        "RANDOM FOREST TRAINING COMPLETE"
    )

    print("=" * 75)

    print(
        "\nBest CV Accuracy:",
        round(
            grid_search.best_score_,
            6
        )
    )

    print(
        "Test Accuracy:",
        round(
            accuracy,
            6
        )
    )

    print(
        f"Top-{TOP_K} Accuracy:",
        round(
            top_k_accuracy,
            6
        )
    )

    print(
        "\nRandom Forest artifacts saved successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
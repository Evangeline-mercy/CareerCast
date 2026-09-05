"""Train CareerCast classifiers on the saved fine-tuned SBERT embeddings.

This script deliberately writes to a new artifact directory.  It does not
modify the currently deployed pretrained-SBERT models.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
CV_FOLDS = 2
TUNING_SAMPLES_PER_CLASS = 40

DATASET_PATH = Path("results/milestone2_training/career_profile_training_dataset.csv")
EMBEDDINGS_PATH = Path(
    "results/milestone2_sentence_bert_finetuned/"
    "sentence_bert_embeddings_finetuned.npy"
)
SPLIT_PATH = Path("results/milestone2_sentence_bert/train_test_split_indices.npz")
OUTPUT_DIR = Path("results/milestone2_finetuned_sbert_classifiers")


def top_k_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    """Return multiclass Top-K accuracy without depending on label ordering helpers."""
    k = min(k, probabilities.shape[1])
    top_k = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_k == y_true[:, None], axis=1)))


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, np.ndarray]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "top1_accuracy": top_k_accuracy(y_test, probabilities, 1),
        "top3_accuracy": top_k_accuracy(y_test, probabilities, 3),
        "top5_accuracy": top_k_accuracy(y_test, probabilities, 5),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }
    return metrics, probabilities


def checked_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder, pd.DataFrame]:
    required = (DATASET_PATH, EMBEDDINGS_PATH, SPLIT_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    frame = pd.read_csv(DATASET_PATH)
    if not {"skills", "career"}.issubset(frame.columns):
        raise ValueError("Training dataset must contain skills and career columns")
    if frame[["skills", "career"]].isna().any(axis=None):
        raise ValueError(
            "Dataset contains missing skills/career values. Do not drop rows because "
            "that would invalidate the saved split indices."
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    split = np.load(SPLIT_PATH)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)

    if embeddings.shape != (len(frame), 384):
        raise ValueError(
            f"Expected {(len(frame), 384)} embeddings, found {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    if set(train_idx).intersection(test_idx):
        raise ValueError("Saved training and test indices overlap")
    if len(train_idx) + len(test_idx) != len(frame):
        raise ValueError("Saved split does not cover the full dataset exactly once")
    if max(train_idx.max(), test_idx.max()) >= len(frame):
        raise ValueError("Saved split contains an out-of-range index")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Fine-tuned embeddings are not L2 normalized")

    encoder = LabelEncoder()
    encoder.fit(frame["career"].astype(str).str.strip())
    return embeddings, train_idx, test_idx, encoder, frame


def searches(class_count: int, cv: StratifiedKFold) -> dict[str, GridSearchCV]:
    thread_count = max(1, int(os.getenv("CAREERCAST_MODEL_THREADS", "1")))
    return {
        "logistic_regression": GridSearchCV(
            LogisticRegression(
                max_iter=2500,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            ),
            {"C": [1.0, 10.0]},
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            refit=False,
            verbose=2,
        ),
        "random_forest": GridSearchCV(
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=thread_count,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 30],
                "min_samples_leaf": [1],
                "max_features": ["sqrt"],
            },
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            refit=False,
            verbose=2,
        ),
        "xgboost": GridSearchCV(
            XGBClassifier(
                objective="multi:softprob",
                num_class=class_count,
                eval_metric="mlogloss",
                tree_method="hist",
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=thread_count,
            ),
            {"n_estimators": [50, 100], "max_depth": [3, 5]},
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            refit=False,
            verbose=2,
        ),
    }


def balanced_tuning_indices(y_train: np.ndarray) -> np.ndarray:
    """Return a deterministic, balanced subset of the training pool for CV."""
    rng = np.random.default_rng(RANDOM_STATE)
    selected = []
    for label in np.unique(y_train):
        candidates = np.flatnonzero(y_train == label)
        count = min(TUNING_SAMPLES_PER_CLASS, len(candidates))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def main() -> None:
    started = time.time()
    print("CareerCast fine-tuned SBERT classifier training")
    print("The existing deployed model directory will not be modified.\n")

    embeddings, train_idx, test_idx, encoder, frame = checked_inputs()
    labels = encoder.transform(frame["career"].astype(str).str.strip())
    x_train, x_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f"Rows: {len(frame)}")
    print(f"Train/test: {len(train_idx)}/{len(test_idx)}")
    print(f"Classes: {len(encoder.classes_)}")
    print(f"Embedding shape: {embeddings.shape}\n")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_searches = searches(len(encoder.classes_), cv)
    tuning_idx = balanced_tuning_indices(y_train)
    x_tune, y_tune = x_train[tuning_idx], y_train[tuning_idx]
    print(
        f"CV tuning subset: {len(tuning_idx)} balanced training-only samples "
        f"({TUNING_SAMPLES_PER_CLASS} per class)"
    )
    print("Selected estimators will be refitted on all training samples.\n")
    results: dict[str, dict] = {}
    trained_models = {}
    prediction_columns = {
        "row_index": test_idx,
        "actual_career": encoder.inverse_transform(y_test),
    }

    for name, search in model_searches.items():
        print(f"\n{'=' * 72}\nTraining and tuning: {name}\n{'=' * 72}")
        model_started = time.time()
        search.fit(x_tune, y_tune)
        final_model = clone(search.estimator).set_params(**search.best_params_)
        print(f"Refitting {name} on all {len(x_train)} training samples...")
        final_model.fit(x_train, y_train)
        metrics, probabilities = evaluate(final_model, x_test, y_test)
        metrics.update(
            {
                "best_cv_accuracy": float(search.best_score_),
                "best_params": search.best_params_,
                "cv_tuning_samples": int(len(tuning_idx)),
                "final_refit_samples": int(len(x_train)),
                "training_seconds": float(time.time() - model_started),
            }
        )
        results[name] = metrics
        trained_models[name] = final_model

        ordered = np.argsort(probabilities, axis=1)[:, ::-1][:, :5]
        prediction_columns[f"{name}_top1"] = encoder.inverse_transform(ordered[:, 0])
        prediction_columns[f"{name}_top5"] = [
            " | ".join(encoder.inverse_transform(row)) for row in ordered
        ]
        print(json.dumps(metrics, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filenames = {
        "logistic_regression": "logistic_regression_model.pkl",
        "random_forest": "random_forest_model.pkl",
        "xgboost": "xgboost_model.pkl",
    }
    for name, model in trained_models.items():
        joblib.dump(model, OUTPUT_DIR / filenames[name])
    joblib.dump(encoder, OUTPUT_DIR / "label_encoder.pkl")

    pd.DataFrame(prediction_columns).to_csv(
        OUTPUT_DIR / "finetuned_classifier_predictions.csv", index=False
    )
    pd.DataFrame(
        [{"model": name, **values} for name, values in results.items()]
    ).to_csv(OUTPUT_DIR / "finetuned_classifier_metrics.csv", index=False)

    winner = max(results, key=lambda key: results[key]["best_cv_accuracy"])
    summary = {
        "project": "CareerCast",
        "pipeline": "Fine-tuned Sentence-BERT embeddings + classifier comparison",
        "embedding_model": "results/semantic_embeddings/sbert_finetuned",
        "embedding_training_artifact": (
            "results/milestone2_sentence_bert_finetuned/model"
        ),
        "embedding_fine_tuned": True,
        "embedding_dimension": 384,
        "dataset_rows": int(len(frame)),
        "training_samples": int(len(train_idx)),
        "testing_samples": int(len(test_idx)),
        "number_of_career_classes": int(len(encoder.classes_)),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "cv_tuning_samples": int(len(tuning_idx)),
        "cv_tuning_samples_per_class": TUNING_SAMPLES_PER_CLASS,
        "final_models_refitted_on_all_training_samples": True,
        "model_selection_metric": "cross_validation_accuracy",
        "test_set_used_for_model_selection": False,
        "best_model": winner,
        "all_models": results,
        "total_time_seconds": float(time.time() - started),
    }
    with (OUTPUT_DIR / "finetuned_classifier_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    metadata = {
        "sbert_model": "results/semantic_embeddings/sbert_finetuned",
        "embedding_dimension": 384,
        "num_classes": int(len(encoder.classes_)),
        "training_rows": int(len(train_idx)),
        "models": list(trained_models),
        "normalized_embeddings": True,
        "split_path": str(SPLIT_PATH).replace("\\", "/"),
    }
    joblib.dump(metadata, OUTPUT_DIR / "metadata.pkl")
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    expected = [OUTPUT_DIR / name for name in filenames.values()] + [
        OUTPUT_DIR / "label_encoder.pkl",
        OUTPUT_DIR / "finetuned_classifier_summary.json",
        OUTPUT_DIR / "metadata.pkl",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Training finished but artifacts are missing: {missing}")

    print(f"\nCompleted. Artifacts saved to: {OUTPUT_DIR}")
    print(f"Selected by CV accuracy: {winner}")
    print("The API has not been changed. Verify these results before integration.")


if __name__ == "__main__":
    main()

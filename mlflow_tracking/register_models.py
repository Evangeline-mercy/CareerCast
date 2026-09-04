"""Track and register CareerCast Milestone 2 models for Milestone 3.

This script does not retrain any model. It reads the saved classifiers and the
verified metric summary, logs them to a local SQLite-backed MLflow store, and
registers one version of each model. A completion marker prevents accidental
duplicate registration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from mlflow.models import infer_signature


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "results" / "milestone2_sentence_bert_classifier"
SUMMARY_PATH = MODEL_DIR / "sbert_classifier_summary.json"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
DB_PATH = PROJECT_ROOT / "mlflow.db"
ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
MARKER_PATH = PROJECT_ROOT / "mlflow_tracking" / "registration_complete.json"

EXPERIMENT_NAME = "CareerCast-Milestone3"

MODEL_CONFIG = {
    "LogisticRegression": {
        "file": "logistic_regression_model.pkl",
        "registered_name": "CareerCast-LogisticRegression",
        "flavor": "sklearn",
    },
    "RandomForestClassifier": {
        "file": "random_forest_model.pkl",
        "registered_name": "CareerCast-RandomForest",
        "flavor": "sklearn",
    },
    "XGBClassifier": {
        "file": "xgboost_model.pkl",
        "registered_name": "CareerCast-XGBoost",
        "flavor": "xgboost",
    },
}


def require_files() -> None:
    required = [SUMMARY_PATH, LABEL_ENCODER_PATH]
    required.extend(MODEL_DIR / config["file"] for config in MODEL_CONFIG.values())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required artifacts are missing:\n" + "\n".join(missing))


def load_summary() -> dict:
    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if "all_models" not in summary:
        raise ValueError("Metric summary does not contain 'all_models'")
    return summary


def metric_record(summary: dict, model_key: str) -> dict:
    for record in summary["all_models"]:
        if record.get("model") == model_key:
            return record
    raise ValueError(f"No metrics found for {model_key}")


def tracking_uri() -> str:
    return f"sqlite:///{DB_PATH.as_posix()}"


def log_registered_model(model, flavor: str, registered_name: str):
    input_example = np.zeros((1, 384), dtype=np.float32)
    prediction_example = model.predict(input_example)
    signature = infer_signature(input_example, prediction_example)

    common = {
        "name": "model",
        "input_example": input_example,
        "signature": signature,
        "registered_model_name": registered_name,
        "await_registration_for": 120,
    }
    if flavor == "xgboost":
        return mlflow.xgboost.log_model(xgb_model=model, **common)
    return mlflow.sklearn.log_model(sk_model=model, **common)


def register_all(force: bool = False) -> dict:
    if MARKER_PATH.exists() and not force:
        marker = json.loads(MARKER_PATH.read_text(encoding="utf-8"))
        print("Models were already registered successfully.")
        print(json.dumps(marker, indent=2))
        print("No duplicate MLflow runs or model versions were created.")
        return marker

    require_files()
    summary = load_summary()
    ARTIFACT_ROOT.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri())
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    completed_runs = []

    for model_key, config in MODEL_CONFIG.items():
        record = metric_record(summary, model_key)
        model_path = MODEL_DIR / config["file"]
        model = joblib.load(model_path)

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=f"CareerCast-{model_key}",
        ) as run:
            mlflow.log_params(
                {
                    "classifier": model_key,
                    "embedding_model": summary["embedding_model"],
                    "embedding_dimension": summary["embedding_dimension"],
                    "embedding_fine_tuned": summary["embedding_fine_tuned"],
                    "training_samples": summary["training_samples"],
                    "testing_samples": summary["testing_samples"],
                    "career_classes": summary["number_of_career_classes"],
                    "random_state": summary["random_state"],
                    "cv_folds": summary["cv_folds"],
                    "selected_best_model": record["selected"],
                }
            )
            for name, value in record.get("best_params", {}).items():
                mlflow.log_param(f"model_{name}", value)

            mlflow.log_metrics(
                {
                    "cv_accuracy": float(record["cv_accuracy"]),
                    "test_accuracy": float(record["test_accuracy"]),
                    "precision_macro": float(record["precision_macro"]),
                    "recall_macro": float(record["recall_macro"]),
                    "macro_f1": float(record["macro_f1"]),
                }
            )
            mlflow.log_artifact(str(SUMMARY_PATH), artifact_path="metadata")
            mlflow.log_artifact(str(LABEL_ENCODER_PATH), artifact_path="preprocessing")
            model_info = log_registered_model(
                model=model,
                flavor=config["flavor"],
                registered_name=config["registered_name"],
            )
            mlflow.set_tags(
                {
                    "project": "CareerCast",
                    "milestone": "Milestone 3",
                    "source_milestone": "Milestone 2",
                    "metric_scope": "internal held-out dataset",
                    "registered_model": config["registered_name"],
                }
            )
            completed_runs.append(
                {
                    "model": config["registered_name"],
                    "run_id": run.info.run_id,
                    "model_uri": model_info.model_uri,
                }
            )
            print(f"Registered {config['registered_name']} from run {run.info.run_id}")

    result = {
        "status": "complete",
        "experiment": EXPERIMENT_NAME,
        "tracking_uri": tracking_uri(),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": completed_runs,
    }
    MARKER_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nAll CareerCast models were tracked and registered successfully.")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create new runs and registered versions even if registration already completed.",
    )
    arguments = parser.parse_args()
    register_all(force=arguments.force)

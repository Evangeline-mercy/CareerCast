"""
CareerCast - Milestone 2
Clean Live Prediction Pipeline

IMPORTANT:
This module uses ONLY the verified 96-class Milestone 2 artifacts.

Pipeline:
Resume text
    -> all-MiniLM-L6-v2
    -> 384-dimensional normalized embedding
    -> LR / RF / XGBoost
    -> predict_proba()
    -> label encoder
    -> Top-K careers

This module intentionally does NOT use:
- old 878-class models
- old career embeddings
- final_recommendations CSV
- v4_output
- old TF-IDF/SVD/RIASEC pipeline
- old hybrid recommender
"""

from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "results"
    / "milestone2_sentence_bert_classifier_v2"
)

LR_PATH = ARTIFACT_DIR / "logistic_regression_model.pkl"
RF_PATH = ARTIFACT_DIR / "random_forest_model.pkl"
XGB_PATH = ARTIFACT_DIR / "xgboost_model.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

EXPECTED_EMBEDDING_DIM = 384
EXPECTED_CLASS_COUNT = 96


# ============================================================
# ARTIFACT VALIDATION
# ============================================================

def validate_artifact_files():
    """Verify that all required artifacts exist."""

    required_files = {
        "Logistic Regression": LR_PATH,
        "Random Forest": RF_PATH,
        "XGBoost": XGB_PATH,
        "Label Encoder": LABEL_ENCODER_PATH,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing verified Milestone 2 artifacts:\n"
            + "\n".join(missing)
        )


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """
    Load ONLY the verified Milestone 2 models.
    """

    validate_artifact_files()

    lr = joblib.load(LR_PATH)
    rf = joblib.load(RF_PATH)
    xgb = joblib.load(XGB_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return lr, rf, xgb, label_encoder


# ============================================================
# MODEL COMPATIBILITY VALIDATION
# ============================================================

def validate_models(lr, rf, xgb, label_encoder):
    """
    Validate that the classifiers belong to the verified
    96-class Milestone 2 pipeline.
    """

    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb,
    }

    # --------------------------------------------------------
    # Label encoder
    # --------------------------------------------------------

    if not hasattr(label_encoder, "classes_"):
        raise ValueError(
            "label_encoder.pkl does not contain classes_."
        )

    encoder_classes = np.asarray(
        label_encoder.classes_
    )

    if len(encoder_classes) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Label encoder has {len(encoder_classes)} "
            f"classes. Expected {EXPECTED_CLASS_COUNT}."
        )

    # --------------------------------------------------------
    # Each classifier
    # --------------------------------------------------------

    expected_class_ids = np.arange(
        EXPECTED_CLASS_COUNT
    )

    for name, model in models.items():

        if not hasattr(model, "classes_"):
            raise ValueError(
                f"{name} does not expose classes_."
            )

        model_classes = np.asarray(
            model.classes_
        )

        if len(model_classes) != EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"{name} has {len(model_classes)} classes. "
                f"Expected {EXPECTED_CLASS_COUNT}."
            )

        if not np.array_equal(
            model_classes,
            expected_class_ids
        ):
            raise ValueError(
                f"{name} class IDs do not match "
                f"0..{EXPECTED_CLASS_COUNT - 1}."
            )

    # --------------------------------------------------------
    # Encoder compatibility
    # --------------------------------------------------------

    if not np.array_equal(
        np.arange(len(encoder_classes)),
        expected_class_ids,
    ):
        raise ValueError(
            "Label encoder class indexing is incompatible."
        )

    return True


# ============================================================
# SBERT
# ============================================================

def load_sbert():
    """
    Load the verified pretrained Sentence-BERT model.

    IMPORTANT:
    This is NOT the old fine-tuned career embedding model.
    """

    model = SentenceTransformer(
        SBERT_MODEL_NAME
    )

    return model


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embedding(
    resume_text,
    sbert_model,
):
    """
    Convert resume text into the verified
    384-dimensional normalized embedding.
    """

    if not isinstance(resume_text, str):
        raise TypeError(
            "resume_text must be a string."
        )

    resume_text = resume_text.strip()

    if not resume_text:
        raise ValueError(
            "Resume text is empty."
        )

    embedding = sbert_model.encode(
        [resume_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    # Expected shape: (1, 384)
    if embedding.shape != (
        1,
        EXPECTED_EMBEDDING_DIM,
    ):
        raise ValueError(
            "Invalid SBERT embedding shape: "
            f"{embedding.shape}. "
            f"Expected (1, {EXPECTED_EMBEDDING_DIM})."
        )

    # Verify normalization
    norm = np.linalg.norm(
        embedding[0]
    )

    if not np.isclose(
        norm,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            f"Embedding is not normalized. "
            f"Norm={norm}"
        )

    return embedding


# ============================================================
# TOP-K DECODING
# ============================================================

def decode_top_k(
    probabilities,
    label_encoder,
    top_k=5,
):
    """
    Convert class probabilities into ranked careers.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probability vector must be one-dimensional."
        )

    if len(probabilities) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Probability vector contains "
            f"{len(probabilities)} classes. "
            f"Expected {EXPECTED_CLASS_COUNT}."
        )

    top_k = min(
        int(top_k),
        EXPECTED_CLASS_COUNT,
    )

    class_ids = np.arange(
        EXPECTED_CLASS_COUNT
    )

    sorted_indices = np.argsort(
        probabilities
    )[::-1][:top_k]

    selected_ids = class_ids[
        sorted_indices
    ]

    career_names = (
        label_encoder.inverse_transform(
            selected_ids
        )
    )

    results = []

    for rank, index in enumerate(
        sorted_indices,
        start=1,
    ):
        results.append(
            {
                "rank": rank,
                "career": str(
                    career_names[rank - 1]
                ),
                "probability": float(
                    probabilities[index]
                ),
            }
        )

    return results


# ============================================================
# SINGLE MODEL PREDICTION
# ============================================================

def predict_single_model(
    model,
    embedding,
    label_encoder,
    top_k=5,
):
    """
    Run predict_proba() for one classifier.
    """

    probabilities = model.predict_proba(
        embedding
    )[0]

    return decode_top_k(
        probabilities,
        label_encoder,
        top_k=top_k,
    )


# ============================================================
# COMPLETE LIVE PREDICTION
# ============================================================

def predict_resume(
    resume_text,
    top_k=5,
):
    """
    Complete verified Milestone 2 prediction.

    Returns:
        {
            "embedding_dimension": 384,
            "models": {
                "Logistic Regression": [...],
                "Random Forest": [...],
                "XGBoost": [...]
            }
        }
    """

    # --------------------------------------------------------
    # Load verified artifacts
    # --------------------------------------------------------

    lr, rf, xgb, label_encoder = (
        load_models()
    )

    # --------------------------------------------------------
    # Validate compatibility
    # --------------------------------------------------------

    validate_models(
        lr,
        rf,
        xgb,
        label_encoder,
    )

    # --------------------------------------------------------
    # Load verified SBERT
    # --------------------------------------------------------

    sbert_model = load_sbert()

    # --------------------------------------------------------
    # Generate ONE embedding
    # --------------------------------------------------------

    embedding = generate_embedding(
        resume_text,
        sbert_model,
    )

    # --------------------------------------------------------
    # SAME embedding -> all three models
    # --------------------------------------------------------

    lr_results = predict_single_model(
        lr,
        embedding,
        label_encoder,
        top_k=top_k,
    )

    rf_results = predict_single_model(
        rf,
        embedding,
        label_encoder,
        top_k=top_k,
    )

    xgb_results = predict_single_model(
        xgb,
        embedding,
        label_encoder,
        top_k=top_k,
    )

    return {
        "embedding_dimension": embedding.shape[1],
        "models": {
            "Logistic Regression": lr_results,
            "Random Forest": rf_results,
            "XGBoost": xgb_results,
        },
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    TEST_TEXT = """
    Python, TensorFlow, SQL, Machine Learning,
    Pandas, NumPy, Deep Learning
    """

    print("=" * 70)
    print("CAREERCAST - CLEAN MILESTONE 2 PREDICTOR TEST")
    print("=" * 70)

    result = predict_resume(
        TEST_TEXT,
        top_k=5,
    )

    print(
        f"\nEmbedding dimension: "
        f"{result['embedding_dimension']}"
    )

    for model_name, predictions in (
        result["models"].items()
    ):

        print("\n" + "=" * 70)
        print(model_name.upper())
        print("=" * 70)

        for item in predictions:

            print(
                f"{item['rank']:2d}. "
                f"{item['career']:<55} "
                f"{item['probability'] * 100:.4f}%"
            )

    print("\nTEST PASSED.")
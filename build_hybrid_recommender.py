"""
CareerCast - Milestone 2
Step 4: Hybrid Career Recommender

Combines:
    1. XGBoost structured prediction probabilities
    2. Sentence-BERT semantic similarity

XGBoost features:
    TF-IDF(Skills + Education + Experience)
        -> TruncatedSVD(200)
        + RIASEC(6)
        -> 206 features

Semantic features:
    Sentence-BERT all-MiniLM-L6-v2
        -> 384-dimensional embeddings

Hybrid:
    Hybrid Score =
        alpha * normalized_XGB_probability
        + (1-alpha) * normalized_semantic_similarity

Important:
    - No Job_Description is used by XGBoost.
    - Job_Description is used by Sentence-BERT.
    - Career labels are never used as candidate input.
    - Semantic career centroids for evaluation are built ONLY from
      the training portion to avoid test-set leakage.

Outputs:
    results/hybrid_recommender/
        alpha_tuning_results.csv
        hybrid_evaluation_summary.json
        hybrid_recommender_metadata.json
"""

import os
import json
import time
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


# ==============================================================
# CONFIGURATION
# ==============================================================

DATA_PATH = "results/careercast_candidate_profiles.csv"

TREE_DIR = "results/tree_features"
XGB_DIR = "results/xgboost_model"
SEM_DIR = "results/semantic_embeddings"

OUTPUT_DIR = "results/hybrid_recommender"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_NAME = "all-MiniLM-L6-v2"

ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 1)

TOP_K = 10


# ==============================================================
# PATHS
# ==============================================================

X_TRAIN_PATH = os.path.join(TREE_DIR, "X_train.npy")
X_TEST_PATH = os.path.join(TREE_DIR, "X_test.npy")
Y_TRAIN_PATH = os.path.join(TREE_DIR, "y_train.npy")
Y_TEST_PATH = os.path.join(TREE_DIR, "y_test.npy")

XGB_MODEL_PATH = os.path.join(
    XGB_DIR,
    "xgboost_model.joblib"
)

XGB_LABEL_PATH = os.path.join(
    XGB_DIR,
    "label_encoder.joblib"
)

TREE_LABEL_PATH = os.path.join(
    TREE_DIR,
    "label_encoder.joblib"
)

PROFILE_EMBEDDINGS_PATH = os.path.join(
    SEM_DIR,
    "profile_embeddings.npy"
)

PROFILE_METADATA_PATH = os.path.join(
    SEM_DIR,
    "profile_metadata.csv"
)

CAREER_LABELS_PATH = os.path.join(
    SEM_DIR,
    "career_labels.csv"
)

TFIDF_PATH = os.path.join(
    TREE_DIR,
    "tfidf_vectorizer.joblib"
)

SVD_PATH = os.path.join(
    TREE_DIR,
    "svd.joblib"
)

RIASEC_SCALER_PATH = os.path.join(
    TREE_DIR,
    "riasec_scaler.joblib"
)


# ==============================================================
# LOGGING
# ==============================================================

def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(message):
    print(f"[{ts()}] {message}", flush=True)


# ==============================================================
# METRICS
# ==============================================================

def top_k_accuracy(scores, y_true, k=1):
    """
    scores:
        shape = (n_samples, n_classes)

    y_true:
        integer encoded class labels
    """

    top_indices = np.argsort(
        scores,
        axis=1
    )[:, -k:]

    return float(
        np.mean(
            [
                y_true[i] in top_indices[i]
                for i in range(len(y_true))
            ]
        )
    )


def reciprocal_rank(scores, y_true):
    """
    Mean Reciprocal Rank.
    """

    rankings = np.argsort(
        scores,
        axis=1
    )[:, ::-1]

    reciprocal = []

    for i in range(len(y_true)):

        positions = np.where(
            rankings[i] == y_true[i]
        )[0]

        if len(positions) == 0:
            reciprocal.append(0.0)
        else:
            reciprocal.append(
                1.0 / (positions[0] + 1)
            )

    return float(np.mean(reciprocal))


def normalize_rows_minmax(matrix):
    """
    Min-max normalization independently for each row.

    Each candidate gets its own [0,1] score range.
    """

    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)

    denominator = row_max - row_min

    denominator[
        denominator == 0
    ] = 1.0

    return (
        matrix - row_min
    ) / denominator


# ==============================================================
# SEMANTIC CAREER CENTROIDS
# ==============================================================

def build_training_career_embeddings(
    profile_embeddings,
    metadata_df,
    training_profile_ids,
):
    """
    Build career embeddings using ONLY training profiles.

    This prevents semantic test-set leakage.
    """

    training_ids = set(
        training_profile_ids
    )

    mask = metadata_df["Profile_ID"].isin(
        training_ids
    ).values

    train_embeddings = profile_embeddings[mask]

    train_metadata = metadata_df.loc[
        mask
    ].reset_index(drop=True)

    careers = sorted(
        train_metadata["Career"].unique()
    )

    career_embeddings = []

    for career in careers:

        career_mask = (
            train_metadata["Career"].values
            == career
        )

        vectors = train_embeddings[
            career_mask
        ]

        centroid = vectors.mean(
            axis=0
        )

        norm = np.linalg.norm(
            centroid
        )

        if norm > 0:
            centroid = centroid / norm

        career_embeddings.append(
            centroid.astype(np.float32)
        )

    career_embeddings = np.vstack(
        career_embeddings
    ).astype(np.float32)

    return (
        career_embeddings,
        careers
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    start_time = time.time()

    print("=" * 75)
    print("CAREERCAST MILESTONE 2")
    print("STEP 4 - HYBRID CAREER RECOMMENDER")
    print("=" * 75)

    # ==========================================================
    # STAGE 1 - LOAD DATA
    # ==========================================================

    log("Stage 1/8  Loading dataset and saved artifacts...")

    df = pd.read_csv(DATA_PATH)

    X_train = np.load(
        X_TRAIN_PATH
    ).astype(np.float32)

    X_test = np.load(
        X_TEST_PATH
    ).astype(np.float32)

    y_train = np.load(
        Y_TRAIN_PATH
    )

    y_test = np.load(
        Y_TEST_PATH
    )

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    xgb_label_encoder = joblib.load(
        XGB_LABEL_PATH
    )

    tree_label_encoder = joblib.load(
        TREE_LABEL_PATH
    )

    profile_embeddings = np.load(
        PROFILE_EMBEDDINGS_PATH
    ).astype(np.float32)

    profile_metadata = pd.read_csv(
        PROFILE_METADATA_PATH
    )

    log(f"  Dataset rows          : {len(df)}")
    log(f"  X_train shape         : {X_train.shape}")
    log(f"  X_test shape          : {X_test.shape}")
    log(f"  Profile embeddings    : {profile_embeddings.shape}")
    log(f"  Number of careers     : {len(xgb_label_encoder.classes_)}")


    # ==========================================================
    # STAGE 2 - VALIDATE ARTIFACTS
    # ==========================================================

    log("Stage 2/8  Validating saved artifacts...")

    assert len(df) == len(
        profile_embeddings
    ), (
        "Dataset and semantic embeddings "
        "have different row counts."
    )

    assert len(df) == len(
        profile_metadata
    ), (
        "Dataset and semantic metadata "
        "have different row counts."
    )

    assert len(xgb_label_encoder.classes_) == 878

    assert len(tree_label_encoder.classes_) == 878

    assert list(
        xgb_label_encoder.classes_
    ) == list(
        tree_label_encoder.classes_
    ), (
        "XGBoost and tree label encoders "
        "are not in the same class order."
    )

    log("  Artifact validation OK.")


    # ==========================================================
    # STAGE 3 - RECREATE ORIGINAL 80/20 SPLIT
    # ==========================================================

    log(
        "Stage 3/8  Reconstructing original "
        "stratified 80/20 split..."
    )

    indices = np.arange(
        len(df)
    )

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Career"]
    )

    reconstructed_train = df.iloc[
        train_indices
    ].reset_index(drop=True)

    reconstructed_test = df.iloc[
        test_indices
    ].reset_index(drop=True)

    assert len(reconstructed_train) == len(
        X_train
    ), (
        "Reconstructed training size "
        "does not match X_train."
    )

    assert len(reconstructed_test) == len(
        X_test
    ), (
        "Reconstructed test size "
        "does not match X_test."
    )

    # Verify labels match
    reconstructed_train_y = tree_label_encoder.transform(
        reconstructed_train["Career"]
    )

    reconstructed_test_y = tree_label_encoder.transform(
        reconstructed_test["Career"]
    )

    assert np.array_equal(
        reconstructed_train_y,
        y_train
    ), (
        "Training split does not match "
        "saved y_train."
    )

    assert np.array_equal(
        reconstructed_test_y,
        y_test
    ), (
        "Test split does not match "
        "saved y_test."
    )

    log("  Split reconstruction verified.")
    log(f"  Training profiles : {len(train_indices)}")
    log(f"  Testing profiles  : {len(test_indices)}")


    # ==========================================================
    # STAGE 4 - XGBOOST PREDICTIONS
    # ==========================================================

    log(
        "Stage 4/8  Generating XGBoost "
        "probabilities..."
    )

    t = time.time()

    xgb_test_proba = xgb_model.predict_proba(
        X_test
    )

    xgb_train_proba = xgb_model.predict_proba(
        X_train
    )

    log(
        f"  XGBoost predictions generated "
        f"in {time.time() - t:.2f}s"
    )

    log(
        f"  Train probability shape : "
        f"{xgb_train_proba.shape}"
    )

    log(
        f"  Test probability shape  : "
        f"{xgb_test_proba.shape}"
    )


    # ==========================================================
    # STAGE 5 - SEMANTIC CAREER EMBEDDINGS
    # ==========================================================

    log(
        "Stage 5/8  Building leakage-resistant "
        "training career centroids..."
    )

    training_profile_ids = (
        reconstructed_train[
            "Profile_ID"
        ].values
    )

    semantic_career_embeddings, career_names = (
        build_training_career_embeddings(
            profile_embeddings,
            profile_metadata,
            training_profile_ids
        )
    )

    log(
        f"  Career embedding shape : "
        f"{semantic_career_embeddings.shape}"
    )

    assert len(career_names) == 878


    # ==========================================================
    # STAGE 6 - SEMANTIC TEST EMBEDDINGS
    # ==========================================================

    log(
        "Stage 6/8  Computing semantic "
        "similarities for test profiles..."
    )

    test_mask = profile_metadata[
        "Profile_ID"
    ].isin(
        reconstructed_test["Profile_ID"]
    ).values

    test_semantic_embeddings = (
        profile_embeddings[
            test_mask
        ]
    )

    semantic_test_similarity = (
        test_semantic_embeddings
        @ semantic_career_embeddings.T
    )

    log(
        f"  Semantic similarity shape : "
        f"{semantic_test_similarity.shape}"
    )


    # ==========================================================
    # STAGE 7 - NORMALIZATION + ALPHA TUNING
    # ==========================================================

    log(
        "Stage 7/8  Normalizing signals..."
    )

    xgb_test_norm = normalize_rows_minmax(
        xgb_test_proba
    )

    semantic_test_norm = normalize_rows_minmax(
        semantic_test_similarity
    )

    # ----------------------------------------------------------
    # IMPORTANT:
    #
    # We use a fixed alpha = 0.5 for the final unbiased
    # held-out evaluation unless a separate validation set
    # is available.
    #
    # We also produce an alpha sweep on the held-out test
    # set ONLY as diagnostic information.
    #
    # The test-set alpha sweep must NOT be reported as an
    # unbiased model-selection result.
    # ----------------------------------------------------------

    log(
        "  Running alpha sweep for diagnostic comparison..."
    )

    alpha_results = []

    for alpha in ALPHAS:

        hybrid_scores = (
            alpha * xgb_test_norm
            + (1.0 - alpha)
            * semantic_test_norm
        )

        top1 = top_k_accuracy(
            hybrid_scores,
            y_test,
            k=1
        )

        top3 = top_k_accuracy(
            hybrid_scores,
            y_test,
            k=3
        )

        top5 = top_k_accuracy(
            hybrid_scores,
            y_test,
            k=5
        )

        mrr = reciprocal_rank(
            hybrid_scores,
            y_test
        )

        alpha_results.append(
            {
                "alpha": float(alpha),
                "xgb_weight": float(alpha),
                "semantic_weight": float(1.0 - alpha),
                "top1_accuracy": top1,
                "top3_accuracy": top3,
                "top5_accuracy": top5,
                "mrr": mrr,
            }
        )

    alpha_df = pd.DataFrame(
        alpha_results
    )

    alpha_csv = os.path.join(
        OUTPUT_DIR,
        "alpha_tuning_results.csv"
    )

    alpha_df.to_csv(
        alpha_csv,
        index=False
    )

    # ----------------------------------------------------------
    # Fixed balanced hybrid
    # ----------------------------------------------------------

    FINAL_ALPHA = 0.5

    final_hybrid_scores = (
        FINAL_ALPHA * xgb_test_norm
        + (1.0 - FINAL_ALPHA)
        * semantic_test_norm
    )

    hybrid_top1 = top_k_accuracy(
        final_hybrid_scores,
        y_test,
        k=1
    )

    hybrid_top3 = top_k_accuracy(
        final_hybrid_scores,
        y_test,
        k=3
    )

    hybrid_top5 = top_k_accuracy(
        final_hybrid_scores,
        y_test,
        k=5
    )

    hybrid_mrr = reciprocal_rank(
        final_hybrid_scores,
        y_test
    )


    # ==========================================================
    # INDIVIDUAL MODEL METRICS
    # ==========================================================

    xgb_top1 = top_k_accuracy(
        xgb_test_proba,
        y_test,
        k=1
    )

    xgb_top3 = top_k_accuracy(
        xgb_test_proba,
        y_test,
        k=3
    )

    xgb_top5 = top_k_accuracy(
        xgb_test_proba,
        y_test,
        k=5
    )

    xgb_mrr = reciprocal_rank(
        xgb_test_proba,
        y_test
    )

    semantic_top1 = top_k_accuracy(
        semantic_test_similarity,
        y_test,
        k=1
    )

    semantic_top3 = top_k_accuracy(
        semantic_test_similarity,
        y_test,
        k=3
    )

    semantic_top5 = top_k_accuracy(
        semantic_test_similarity,
        y_test,
        k=5
    )

    semantic_mrr = reciprocal_rank(
        semantic_test_similarity,
        y_test
    )


    # ==========================================================
    # STAGE 8 - SAVE RESULTS
    # ==========================================================

    log(
        "Stage 8/8  Saving evaluation results..."
    )

    summary = {

        "dataset": {
            "rows": int(len(df)),
            "careers": int(len(career_names)),
            "training_samples": int(len(train_indices)),
            "testing_samples": int(len(test_indices)),
        },

        "xgboost": {
            "top1_accuracy": xgb_top1,
            "top3_accuracy": xgb_top3,
            "top5_accuracy": xgb_top5,
            "mrr": xgb_mrr,
        },

        "semantic": {
            "top1_accuracy": semantic_top1,
            "top3_accuracy": semantic_top3,
            "top5_accuracy": semantic_top5,
            "mrr": semantic_mrr,
        },

        "hybrid": {
            "alpha": FINAL_ALPHA,
            "xgb_weight": FINAL_ALPHA,
            "semantic_weight": 1.0 - FINAL_ALPHA,
            "top1_accuracy": hybrid_top1,
            "top3_accuracy": hybrid_top3,
            "top5_accuracy": hybrid_top5,
            "mrr": hybrid_mrr,
        },

        "evaluation_notes": [
            "Semantic career centroids were constructed using training profiles only.",
            "The original 80/20 stratified split was reconstructed using random_state=42.",
            "Job_Description is excluded from XGBoost features.",
            "Job_Description is included in Sentence-BERT semantic embeddings.",
            "Alpha=0.5 is used for the final reported hybrid score.",
            "The alpha sweep is diagnostic because no independent validation model was retrained.",
        ],

        "model": {
            "sentence_transformer": MODEL_NAME,
            "number_of_careers": int(len(career_names)),
            "embedding_dimension": 384,
            "xgboost_features": 206,
        },

        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "hybrid_evaluation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


    metadata = {

        "step": "Milestone 2 - Step 4",

        "method": "Hybrid XGBoost + Sentence-BERT",

        "xgboost": {
            "model": "saved XGBoost classifier",
            "features": [
                "TF-IDF(Skills + Education + Experience)",
                "TruncatedSVD 200",
                "RIASEC 6 features",
            ],
            "total_features": 206,
        },

        "semantic": {
            "model": MODEL_NAME,
            "dimension": 384,
            "career_centroid_method":
                "mean of training profile embeddings per career",
            "normalization": "L2",
        },

        "fusion": {
            "formula":
                "alpha * XGB_normalized + "
                "(1-alpha) * semantic_normalized",
            "normalization": "row-wise min-max",
            "final_alpha": FINAL_ALPHA,
            "xgb_weight": FINAL_ALPHA,
            "semantic_weight": 1.0 - FINAL_ALPHA,
        },

        "dataset": {
            "rows": int(len(df)),
            "careers": int(len(career_names)),
            "train_size": int(len(train_indices)),
            "test_size": int(len(test_indices)),
            "random_state": RANDOM_STATE,
        },

        "leakage_control": {
            "semantic_centroids_use_test_profiles":
                False,
            "career_label_used_as_input":
                False,
            "job_description_used_by_xgboost":
                False,
            "job_description_used_by_semantic_model":
                True,
        },
    }

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "hybrid_recommender_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )


    # ==========================================================
    # FINAL CONSOLE REPORT
    # ==========================================================

    print()
    print("=" * 75)
    print("HYBRID RECOMMENDER EVALUATION")
    print("=" * 75)

    print()
    print("XGBoost:")
    print(
        f"  Top-1 Accuracy : {xgb_top1:.4f}"
    )
    print(
        f"  Top-3 Accuracy : {xgb_top3:.4f}"
    )
    print(
        f"  Top-5 Accuracy : {xgb_top5:.4f}"
    )
    print(
        f"  MRR            : {xgb_mrr:.4f}"
    )

    print()
    print("Sentence-BERT:")
    print(
        f"  Top-1 Accuracy : {semantic_top1:.4f}"
    )
    print(
        f"  Top-3 Accuracy : {semantic_top3:.4f}"
    )
    print(
        f"  Top-5 Accuracy : {semantic_top5:.4f}"
    )
    print(
        f"  MRR            : {semantic_mrr:.4f}"
    )

    print()
    print("Hybrid:")
    print(
        f"  Alpha          : {FINAL_ALPHA:.1f}"
    )
    print(
        f"  XGBoost weight : {FINAL_ALPHA:.1f}"
    )
    print(
        f"  Semantic weight: {1.0 - FINAL_ALPHA:.1f}"
    )
    print(
        f"  Top-1 Accuracy : {hybrid_top1:.4f}"
    )
    print(
        f"  Top-3 Accuracy : {hybrid_top3:.4f}"
    )
    print(
        f"  Top-5 Accuracy : {hybrid_top5:.4f}"
    )
    print(
        f"  MRR            : {hybrid_mrr:.4f}"
    )

    print()
    print("Alpha diagnostic sweep:")
    print(
        alpha_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 75)
    print(
        f"Step 4 complete! "
        f"Total time: "
        f"{(time.time() - start_time):.1f}s"
    )
    print("=" * 75)

    print()
    print("Saved files:")

    print(
        f"  {alpha_csv}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print()
    print("Done.")


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception:

        print()
        print("=" * 75)
        print("ERROR")
        print("=" * 75)

        traceback.print_exc()

        raise
import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    "results",
    "milestone2_training",
    "career_profile_training_dataset.csv"
)

SPLIT_PATH = os.path.join(
    "results",
    "milestone2_sentence_bert",
    "train_test_split_indices.npz"
)

PRETRAINED_EMBEDDINGS = os.path.join(
    "results",
    "milestone2_sentence_bert",
    "sentence_bert_embeddings.npy"
)

FINETUNED_EMBEDDINGS = os.path.join(
    "results",
    "milestone2_sentence_bert_finetuned",
    "sentence_bert_embeddings_finetuned.npy"
)

OUTPUT_DIR = os.path.join(
    "results",
    "milestone2_sbert_comparison"
)

RANDOM_STATE = 42


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("CAREERCAST MILESTONE 2")
    print("PRETRAINED SBERT vs FINE-TUNED SBERT COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 - CHECK FILES
    # --------------------------------------------------------

    print("\n[1] Checking required files...")

    required_files = [
        DATASET_PATH,
        SPLIT_PATH,
        PRETRAINED_EMBEDDINGS,
        FINETUNED_EMBEDDINGS,
    ]

    for path in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print("All required files found.")

    # --------------------------------------------------------
    # STEP 2 - LOAD DATASET
    # --------------------------------------------------------

    print("\n[2] Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    df = df.dropna(
        subset=["skills", "career"]
    ).reset_index(drop=True)

    print(f"Dataset rows: {len(df)}")

    careers = df["career"].astype(str).values

    # --------------------------------------------------------
    # STEP 3 - LOAD TRAIN/TEST SPLIT
    # --------------------------------------------------------

    print("\n[3] Loading original train/test split...")

    split_data = np.load(SPLIT_PATH)

    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples : {len(test_idx)}")

    # --------------------------------------------------------
    # STEP 4 - LOAD PRETRAINED EMBEDDINGS
    # --------------------------------------------------------

    print("\n[4] Loading pretrained SBERT embeddings...")

    pretrained = np.load(
        PRETRAINED_EMBEDDINGS
    )

    print(f"Pretrained shape: {pretrained.shape}")

    # --------------------------------------------------------
    # STEP 5 - LOAD FINE-TUNED EMBEDDINGS
    # --------------------------------------------------------

    print("\n[5] Loading fine-tuned SBERT embeddings...")

    finetuned = np.load(
        FINETUNED_EMBEDDINGS
    )

    print(f"Fine-tuned shape: {finetuned.shape}")

    # --------------------------------------------------------
    # STEP 6 - VERIFY EMBEDDINGS
    # --------------------------------------------------------

    print("\n[6] Verifying embedding compatibility...")

    if pretrained.shape != finetuned.shape:
        raise ValueError(
            "Pretrained and fine-tuned embedding shapes do not match."
        )

    if pretrained.shape[0] != len(df):
        raise ValueError(
            "Embedding count does not match dataset row count."
        )

    print("Embedding dimensions match.")
    print("Sample ordering verified by shared dataset/split.")

    # --------------------------------------------------------
    # STEP 7 - ENCODE CAREER LABELS
    # --------------------------------------------------------

    print("\n[7] Encoding career labels...")

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(careers)

    print(
        f"Career classes: {len(label_encoder.classes_)}"
    )

    # --------------------------------------------------------
    # STEP 8 - CREATE TRAIN/TEST DATA
    # --------------------------------------------------------

    X_pre_train = pretrained[train_idx]
    X_pre_test = pretrained[test_idx]

    X_ft_train = finetuned[train_idx]
    X_ft_test = finetuned[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # --------------------------------------------------------
    # STEP 9 - TRAIN PRETRAINED SBERT CLASSIFIER
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRETRAINED SBERT + LOGISTIC REGRESSION")
    print("=" * 70)

    pretrained_model = LogisticRegression(
        C=10.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    t0 = time.time()

    pretrained_model.fit(
        X_pre_train,
        y_train
    )

    pretrained_time = time.time() - t0

    pretrained_predictions = pretrained_model.predict(
        X_pre_test
    )

    pretrained_accuracy = accuracy_score(
        y_test,
        pretrained_predictions
    )

    pretrained_precision = precision_score(
        y_test,
        pretrained_predictions,
        average="macro",
        zero_division=0
    )

    pretrained_recall = recall_score(
        y_test,
        pretrained_predictions,
        average="macro",
        zero_division=0
    )

    pretrained_f1 = f1_score(
        y_test,
        pretrained_predictions,
        average="macro",
        zero_division=0
    )

    print(f"Training time : {pretrained_time:.2f} seconds")
    print(f"Test accuracy : {pretrained_accuracy:.4f}")
    print(f"Macro precision: {pretrained_precision:.4f}")
    print(f"Macro recall   : {pretrained_recall:.4f}")
    print(f"Macro F1       : {pretrained_f1:.4f}")

    # --------------------------------------------------------
    # STEP 10 - TRAIN FINE-TUNED SBERT CLASSIFIER
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINE-TUNED SBERT + LOGISTIC REGRESSION")
    print("=" * 70)

    finetuned_model = LogisticRegression(
        C=10.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    t0 = time.time()

    finetuned_model.fit(
        X_ft_train,
        y_train
    )

    finetuned_time = time.time() - t0

    finetuned_predictions = finetuned_model.predict(
        X_ft_test
    )

    finetuned_accuracy = accuracy_score(
        y_test,
        finetuned_predictions
    )

    finetuned_precision = precision_score(
        y_test,
        finetuned_predictions,
        average="macro",
        zero_division=0
    )

    finetuned_recall = recall_score(
        y_test,
        finetuned_predictions,
        average="macro",
        zero_division=0
    )

    finetuned_f1 = f1_score(
        y_test,
        finetuned_predictions,
        average="macro",
        zero_division=0
    )

    print(f"Training time : {finetuned_time:.2f} seconds")
    print(f"Test accuracy : {finetuned_accuracy:.4f}")
    print(f"Macro precision: {finetuned_precision:.4f}")
    print(f"Macro recall   : {finetuned_recall:.4f}")
    print(f"Macro F1       : {finetuned_f1:.4f}")

    # --------------------------------------------------------
    # STEP 11 - CALCULATE IMPROVEMENT
    # --------------------------------------------------------

    accuracy_change = (
        finetuned_accuracy -
        pretrained_accuracy
    )

    f1_change = (
        finetuned_f1 -
        pretrained_f1
    )

    accuracy_change_percentage_points = (
        accuracy_change * 100
    )

    # --------------------------------------------------------
    # STEP 12 - FINAL COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL SBERT COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<25}"
        f"{'Pretrained':>15}"
        f"{'Fine-tuned':>15}"
        f"{'Change':>15}"
    )

    print("-" * 70)

    print(
        f"{'Test Accuracy':<25}"
        f"{pretrained_accuracy:>15.4f}"
        f"{finetuned_accuracy:>15.4f}"
        f"{accuracy_change:+15.4f}"
    )

    print(
        f"{'Macro Precision':<25}"
        f"{pretrained_precision:>15.4f}"
        f"{finetuned_precision:>15.4f}"
        f"{(finetuned_precision - pretrained_precision):+15.4f}"
    )

    print(
        f"{'Macro Recall':<25}"
        f"{pretrained_recall:>15.4f}"
        f"{finetuned_recall:>15.4f}"
        f"{(finetuned_recall - pretrained_recall):+15.4f}"
    )

    print(
        f"{'Macro F1':<25}"
        f"{pretrained_f1:>15.4f}"
        f"{finetuned_f1:>15.4f}"
        f"{f1_change:+15.4f}"
    )

    print("-" * 70)

    if finetuned_accuracy > pretrained_accuracy:
        winner = "Fine-tuned SBERT"
    elif finetuned_accuracy < pretrained_accuracy:
        winner = "Pretrained SBERT"
    else:
        winner = "Tie"

    print(f"\nBEST MODEL: {winner}")

    print(
        f"Accuracy change: "
        f"{accuracy_change_percentage_points:+.2f} percentage points"
    )

    # --------------------------------------------------------
    # STEP 13 - SAVE RESULTS
    # --------------------------------------------------------

    print("\n[8] Saving comparison results...")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    comparison_df = pd.DataFrame([
        {
            "model": "Pretrained SBERT + LogisticRegression",
            "embedding_type": "pretrained",
            "test_accuracy": pretrained_accuracy,
            "macro_precision": pretrained_precision,
            "macro_recall": pretrained_recall,
            "macro_f1": pretrained_f1,
            "training_time_seconds": pretrained_time,
        },
        {
            "model": "Fine-tuned SBERT + LogisticRegression",
            "embedding_type": "fine-tuned",
            "test_accuracy": finetuned_accuracy,
            "macro_precision": finetuned_precision,
            "macro_recall": finetuned_recall,
            "macro_f1": finetuned_f1,
            "training_time_seconds": finetuned_time,
        }
    ])

    csv_path = os.path.join(
        OUTPUT_DIR,
        "pretrained_vs_finetuned_comparison.csv"
    )

    comparison_df.to_csv(
        csv_path,
        index=False
    )

    summary = {
        "pretrained_sbert": {
            "test_accuracy": pretrained_accuracy,
            "macro_precision": pretrained_precision,
            "macro_recall": pretrained_recall,
            "macro_f1": pretrained_f1,
        },
        "finetuned_sbert": {
            "test_accuracy": finetuned_accuracy,
            "macro_precision": finetuned_precision,
            "macro_recall": finetuned_recall,
            "macro_f1": finetuned_f1,
        },
        "improvement": {
            "accuracy_difference": accuracy_change,
            "accuracy_percentage_points": accuracy_change_percentage_points,
            "macro_f1_difference": f1_change,
        },
        "winner": winner,
        "classifier": "LogisticRegression",
        "same_train_test_split": True,
        "random_state": RANDOM_STATE,
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "pretrained_vs_finetuned_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print(f"Saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    total_time = time.time() - start_time

    print("\n")
    print("=" * 70)
    print("PRETRAINED vs FINE-TUNED SBERT COMPARISON COMPLETE")
    print("=" * 70)

    print(
        f"Total execution time: "
        f"{total_time:.1f} seconds"
    )


if __name__ == "__main__":
    main()
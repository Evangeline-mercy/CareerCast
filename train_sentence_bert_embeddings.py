import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print()
    print("Install it using:")
    print("pip install sentence-transformers")
    sys.exit(1)


# ==============================================================
# CONFIGURATION
# ==============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

INPUT_PATH = os.path.join(
    "results",
    "milestone2_training",
    "career_profile_training_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    "results",
    "milestone2_sentence_bert"
)

MODEL_NAME = "all-MiniLM-L6-v2"

# 128 is a reasonable batch size.
# If your PC runs out of memory, reduce this to 64 or 32.
BATCH_SIZE = 128

# This script generates PRETRAINED embeddings.
# It does NOT fine-tune SBERT.
IS_FINE_TUNED = False

SCRIPT_START = time.time()


# ==============================================================
# LOGGING
# ==============================================================

def log(step, message):
    elapsed = time.time() - SCRIPT_START
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] (+{elapsed:.1f}s) [{step}] {message}")


def fail(message, exception=None):
    print()
    print("=" * 70)
    print("ERROR: " + message)

    if exception is not None:
        print("-" * 70)
        traceback.print_exc()

    print("=" * 70)
    sys.exit(1)


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print("CAREERCAST MILESTONE 2 - SENTENCE-BERT SKILL EMBEDDINGS")
    print("=" * 70)
    print()

    print(f"Sentence-BERT model: {MODEL_NAME}")
    print("Embedding type: PRETRAINED")
    print("Fine-tuning: NOT performed")
    print()

    # ----------------------------------------------------------
    # STAGE 1 - LOAD DATASET
    # ----------------------------------------------------------

    log("1", "Loading career profile training dataset...")

    try:

        if not os.path.isfile(INPUT_PATH):
            fail(
                f"Dataset not found at:\n"
                f"'{INPUT_PATH}'"
            )

        df = pd.read_csv(INPUT_PATH)

        required_columns = {"skills", "career"}

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            fail(
                f"Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        df = df.dropna(
            subset=["skills", "career"]
        ).reset_index(drop=True)

        if len(df) == 0:
            fail(
                "Dataset contains zero usable rows "
                "after removing missing values."
            )

    except SystemExit:
        raise

    except Exception as e:
        fail(
            f"Failed to load dataset: {e}",
            e
        )

    n_samples = len(df)
    n_careers = df["career"].nunique()

    log(
        "1",
        f"Dataset loaded successfully: "
        f"{n_samples} rows, {n_careers} career classes."
    )

    skills_text = (
        df["skills"]
        .astype(str)
        .tolist()
    )

    career_labels = (
        df["career"]
        .astype(str)
        .tolist()
    )

    # ----------------------------------------------------------
    # STAGE 2 - CREATE TRAIN / TEST SPLIT
    # ----------------------------------------------------------

    log(
        "2",
        "Creating stratified train/test split..."
    )

    try:

        all_indices = np.arange(n_samples)

        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["career"]
        )

    except Exception as e:

        fail(
            f"Stratified train/test split failed: {e}",
            e
        )

    log(
        "2",
        f"Split created: "
        f"{len(train_idx)} training / "
        f"{len(test_idx)} testing samples."
    )

    # ----------------------------------------------------------
    # STAGE 3 - SELECT DEVICE
    # ----------------------------------------------------------

    log(
        "3",
        "Checking available computing device..."
    )

    try:

        if torch is not None and torch.cuda.is_available():

            device = "cuda"

            log(
                "3",
                "CUDA GPU detected. Sentence-BERT will use GPU."
            )

        else:

            device = "cpu"

            log(
                "3",
                "CUDA GPU not available. "
                "Sentence-BERT will use CPU."
            )

    except Exception:

        device = "cpu"

        log(
            "3",
            "Could not detect GPU. Using CPU."
        )

    # ----------------------------------------------------------
    # STAGE 4 - LOAD SENTENCE-BERT
    # ----------------------------------------------------------

    log(
        "4",
        f"Loading Sentence-BERT model '{MODEL_NAME}'..."
    )

    try:

        model = SentenceTransformer(
            MODEL_NAME,
            device=device
        )

        embedding_dimension = (
            model.get_sentence_embedding_dimension()
        )

    except Exception as e:

        fail(
            f"Failed to load Sentence-BERT model "
            f"'{MODEL_NAME}': {e}",
            e
        )

    log(
        "4",
        f"Model loaded successfully."
    )

    log(
        "4",
        f"Device: {device}"
    )

    log(
        "4",
        f"Embedding dimension: {embedding_dimension}"
    )

    # ----------------------------------------------------------
    # STAGE 5 - GENERATE EMBEDDINGS
    # ----------------------------------------------------------

    log(
        "5",
        f"Generating embeddings for "
        f"{n_samples} samples..."
    )

    print()
    print(
        f"Batch size: {BATCH_SIZE}"
    )
    print(
        "This may take several minutes on CPU."
    )
    print()

    try:

        embedding_start = time.time()

        embeddings = model.encode(
            skills_text,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype(
            np.float32,
            copy=False
        )

        embedding_time = (
            time.time() - embedding_start
        )

    except Exception as e:

        fail(
            f"Sentence-BERT embedding generation failed: {e}",
            e
        )

    # ----------------------------------------------------------
    # VALIDATE EMBEDDINGS
    # ----------------------------------------------------------

    if embeddings.shape[0] != n_samples:

        fail(
            f"Embedding count ({embeddings.shape[0]}) "
            f"does not match dataset size ({n_samples})."
        )

    if embeddings.shape[1] != embedding_dimension:

        fail(
            f"Embedding dimension mismatch. "
            f"Expected {embedding_dimension}, "
            f"received {embeddings.shape[1]}."
        )

    log(
        "5",
        f"Embedding generation completed "
        f"in {embedding_time:.1f} seconds."
    )

    log(
        "5",
        f"Embedding matrix shape: {embeddings.shape}"
    )

    # ----------------------------------------------------------
    # STAGE 6 - CREATE OUTPUT DIRECTORY
    # ----------------------------------------------------------

    log(
        "6",
        "Creating output directory..."
    )

    try:

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

    except Exception as e:

        fail(
            f"Could not create output directory: {e}",
            e
        )

    # ----------------------------------------------------------
    # STAGE 7 - DEFINE OUTPUT FILES
    # ----------------------------------------------------------

    embeddings_path = os.path.join(
        OUTPUT_DIR,
        "sentence_bert_embeddings.npy"
    )

    split_path = os.path.join(
        OUTPUT_DIR,
        "train_test_split_indices.npz"
    )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "sentence_bert_metadata.csv"
    )

    model_info_path = os.path.join(
        OUTPUT_DIR,
        "sentence_bert_model_info.json"
    )

    # ----------------------------------------------------------
    # STAGE 8 - SAVE EMBEDDINGS
    # ----------------------------------------------------------

    log(
        "7",
        "Saving Sentence-BERT embeddings..."
    )

    try:

        np.save(
            embeddings_path,
            embeddings
        )

        log(
            "7",
            f"Saved: {embeddings_path}"
        )

    except Exception as e:

        fail(
            f"Failed to save embeddings: {e}",
            e
        )

    # ----------------------------------------------------------
    # STAGE 9 - SAVE TRAIN / TEST SPLIT
    # ----------------------------------------------------------

    log(
        "8",
        "Saving train/test split indices..."
    )

    try:

        np.savez(
            split_path,
            train_idx=train_idx,
            test_idx=test_idx,
            random_state=RANDOM_STATE,
            test_size=TEST_SIZE
        )

        log(
            "8",
            f"Saved: {split_path}"
        )

    except Exception as e:

        fail(
            f"Failed to save split indices: {e}",
            e
        )

    # ----------------------------------------------------------
    # STAGE 10 - SAVE METADATA
    # ----------------------------------------------------------

    log(
        "9",
        "Saving metadata..."
    )

    try:

        split_membership = np.full(
            n_samples,
            "train",
            dtype=object
        )

        split_membership[test_idx] = "test"

        metadata_df = pd.DataFrame({
            "row_index": np.arange(n_samples),
            "career": career_labels,
            "split": split_membership
        })

        metadata_df.to_csv(
            metadata_path,
            index=False
        )

        log(
            "9",
            f"Saved: {metadata_path}"
        )

    except Exception as e:

        fail(
            f"Failed to save metadata: {e}",
            e
        )

    # ----------------------------------------------------------
    # STAGE 11 - SAVE MODEL INFORMATION
    # ----------------------------------------------------------

    log(
        "10",
        "Saving Sentence-BERT model information..."
    )

    total_time = (
        time.time() - SCRIPT_START
    )

    model_info = {

        "model_name": MODEL_NAME,

        "is_fine_tuned": IS_FINE_TUNED,

        "embedding_type":
            "pretrained_sentence_transformer",

        "fine_tuning_note":
            "The Sentence-BERT model was used "
            "as a pretrained embedding model. "
            "No fine-tuning was performed.",

        "embedding_dimension":
            int(embedding_dimension),

        "n_samples":
            int(n_samples),

        "n_careers":
            int(n_careers),

        "n_train":
            int(len(train_idx)),

        "n_test":
            int(len(test_idx)),

        "device_used":
            device,

        "batch_size":
            BATCH_SIZE,

        "normalize_embeddings":
            True,

        "embedding_generation_time_seconds":
            float(embedding_time),

        "total_time_seconds":
            float(total_time),

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "source_dataset":
            INPUT_PATH
    }

    try:

        with open(
            model_info_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                model_info,
                f,
                indent=2
            )

        log(
            "10",
            f"Saved: {model_info_path}"
        )

    except Exception as e:

        fail(
            f"Failed to save model information: {e}",
            e
        )

    # ----------------------------------------------------------
    # STAGE 12 - VERIFY FILES
    # ----------------------------------------------------------

    log(
        "11",
        "Verifying generated files..."
    )

    expected_files = [

        embeddings_path,

        split_path,

        metadata_path,

        model_info_path
    ]

    missing_files = [
        path
        for path in expected_files
        if not os.path.isfile(path)
    ]

    if missing_files:

        fail(
            "The following expected files "
            "were not created:\n"
            + "\n".join(missing_files)
        )

    log(
        "11",
        "All Sentence-BERT files verified successfully."
    )

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("SENTENCE-BERT EMBEDDING GENERATION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "Embedding type: PRETRAINED "
        "(NOT fine-tuned)"
    )

    print(
        f"Device used: {device}"
    )

    print(
        f"Embedding dimension: "
        f"{embedding_dimension}"
    )

    print(
        f"Samples embedded: "
        f"{n_samples}"
    )

    print(
        f"Training samples: "
        f"{len(train_idx)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_idx)}"
    )

    print(
        f"Career classes: "
        f"{n_careers}"
    )

    print(
        f"Embedding generation time: "
        f"{embedding_time:.1f} seconds"
    )

    print(
        f"Total time: "
        f"{total_time:.1f} seconds"
    )

    print()
    print("Saved files:")

    for path in expected_files:
        print(
            f"  {path}"
        )

    print()
    print("=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    print()
    print(
        "Run:"
    )
    print(
        "python train_sentence_bert_classifier.py"
    )
    print()


# ==============================================================
# PROGRAM ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    try:

        main()

    except SystemExit:

        raise

    except Exception as e:

        fail(
            "Unhandled error in main().",
            e
        )
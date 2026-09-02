import os
import sys
import time
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("Run:")
    print("pip install -U sentence-transformers==6.0.0")
    sys.exit(1)


# ================================================================
# CONFIGURATION
# ================================================================

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

FINETUNED_MODEL_PATH = os.path.join(
    "results",
    "milestone2_sentence_bert_finetuned",
    "model"
)

OUTPUT_DIR = os.path.join(
    "results",
    "milestone2_sentence_bert_finetuned"
)

EMBEDDINGS_PATH = os.path.join(
    OUTPUT_DIR,
    "sentence_bert_embeddings_finetuned.npy"
)

METADATA_PATH = os.path.join(
    OUTPUT_DIR,
    "sentence_bert_finetuned_metadata.csv"
)

MODEL_INFO_PATH = os.path.join(
    OUTPUT_DIR,
    "sentence_bert_finetuned_model_info.json"
)

BATCH_SIZE = 64

SCRIPT_START = time.time()


# ================================================================
# LOGGING
# ================================================================

def log(message):
    elapsed = time.time() - SCRIPT_START
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] (+{elapsed:.1f}s) {message}")


def fail(message, exc=None):
    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print(message)

    if exc is not None:
        print("-" * 70)
        traceback.print_exc()

    sys.exit(1)


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("CAREERCAST MILESTONE 2")
    print("FINE-TUNED SENTENCE-BERT EMBEDDING GENERATION")
    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # STAGE 1 - CHECK FILES
    # ------------------------------------------------------------

    log("Checking required files...")

    if not os.path.isfile(DATASET_PATH):
        fail(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    if not os.path.isfile(SPLIT_PATH):
        fail(
            f"Saved train/test split not found:\n{SPLIT_PATH}"
        )

    if not os.path.isdir(FINETUNED_MODEL_PATH):
        fail(
            f"Fine-tuned SBERT model not found:\n{FINETUNED_MODEL_PATH}"
        )

    log("All required files found.")

    # ------------------------------------------------------------
    # STAGE 2 - LOAD DATASET
    # ------------------------------------------------------------

    log("Loading dataset...")

    try:
        df = pd.read_csv(DATASET_PATH)

        required_columns = {"skills", "career"}

        missing = required_columns - set(df.columns)

        if missing:
            fail(
                f"Dataset is missing columns: {sorted(missing)}"
            )

        df = df.dropna(
            subset=["skills", "career"]
        ).reset_index(drop=True)

    except Exception as e:
        fail("Failed to load dataset.", e)

    log(f"Dataset rows: {len(df)}")

    skills = df["skills"].astype(str).tolist()
    careers = df["career"].astype(str).tolist()

    # ------------------------------------------------------------
    # STAGE 3 - LOAD ORIGINAL SPLIT
    # ------------------------------------------------------------

    log("Loading original train/test split...")

    try:
        split_data = np.load(SPLIT_PATH)

        train_idx = split_data["train_idx"]
        test_idx = split_data["test_idx"]

    except Exception as e:
        fail("Failed to load train/test split.", e)

    log(f"Training samples: {len(train_idx)}")
    log(f"Testing samples: {len(test_idx)}")

    # ------------------------------------------------------------
    # STAGE 4 - LOAD FINE-TUNED MODEL
    # ------------------------------------------------------------

    log("Loading fine-tuned Sentence-BERT model...")

    try:

        model = SentenceTransformer(
            FINETUNED_MODEL_PATH,
            device="cpu"
        )

    except Exception as e:
        fail(
            "Failed to load fine-tuned SBERT model.",
            e
        )

    embedding_dimension = model.get_sentence_embedding_dimension()

    log(
        f"Fine-tuned model loaded successfully."
    )

    log(
        f"Embedding dimension: {embedding_dimension}"
    )

    # ------------------------------------------------------------
    # STAGE 5 - GENERATE EMBEDDINGS
    # ------------------------------------------------------------

    log(
        "Generating embeddings for all dataset rows..."
    )

    try:

        start_embedding = time.time()

        embeddings = model.encode(
            skills,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embedding_time = time.time() - start_embedding

    except Exception as e:
        fail(
            "Fine-tuned embedding generation failed.",
            e
        )

    log(
        f"Embedding generation completed in "
        f"{embedding_time:.1f} seconds."
    )

    log(
        f"Embedding shape: {embeddings.shape}"
    )

    # ------------------------------------------------------------
    # STAGE 6 - VERIFY SHAPE
    # ------------------------------------------------------------

    expected_shape = (
        len(df),
        embedding_dimension
    )

    if embeddings.shape != expected_shape:
        fail(
            f"Unexpected embedding shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual: {embeddings.shape}"
        )

    log("Embedding shape verified.")

    # ------------------------------------------------------------
    # STAGE 7 - SAVE EMBEDDINGS
    # ------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    log("Saving fine-tuned embeddings...")

    try:

        np.save(
            EMBEDDINGS_PATH,
            embeddings
        )

    except Exception as e:
        fail(
            "Failed to save embeddings.",
            e
        )

    # ------------------------------------------------------------
    # STAGE 8 - SAVE METADATA
    # ------------------------------------------------------------

    log("Saving metadata...")

    try:

        metadata = pd.DataFrame({
            "index": np.arange(len(df)),
            "career": careers,
            "skills": skills
        })

        metadata["split"] = "train"

        metadata.loc[
            test_idx,
            "split"
        ] = "test"

        metadata.to_csv(
            METADATA_PATH,
            index=False
        )

    except Exception as e:
        fail(
            "Failed to save metadata.",
            e
        )

    # ------------------------------------------------------------
    # STAGE 9 - SAVE MODEL INFORMATION
    # ------------------------------------------------------------

    total_time = time.time() - SCRIPT_START

    model_info = {
        "model_type": "Sentence-BERT",
        "base_model": "all-MiniLM-L6-v2",
        "fine_tuned": True,
        "fine_tuned_model_path": FINETUNED_MODEL_PATH,
        "embedding_dimension": int(embedding_dimension),
        "samples_embedded": int(len(df)),
        "training_samples": int(len(train_idx)),
        "testing_samples": int(len(test_idx)),
        "batch_size": BATCH_SIZE,
        "device": "cpu",
        "normalized_embeddings": True,
        "embedding_generation_time_seconds": float(
            embedding_time
        ),
        "total_time_seconds": float(
            total_time
        ),
        "note": (
            "Embeddings generated using the fine-tuned "
            "Sentence-BERT model trained with label-based "
            "same-career positive pairs."
        )
    }

    try:

        with open(
            MODEL_INFO_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                model_info,
                f,
                indent=2
            )

    except Exception as e:
        fail(
            "Failed to save model information.",
            e
        )

    # ------------------------------------------------------------
    # STAGE 10 - VERIFY OUTPUTS
    # ------------------------------------------------------------

    log("Verifying generated files...")

    required_outputs = [
        EMBEDDINGS_PATH,
        METADATA_PATH,
        MODEL_INFO_PATH
    ]

    missing_outputs = [
        path
        for path in required_outputs
        if not os.path.isfile(path)
    ]

    if missing_outputs:

        fail(
            "The following files were not generated:\n"
            + "\n".join(missing_outputs)
        )

    log("All fine-tuned SBERT files verified successfully.")

    # ------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINE-TUNED SBERT EMBEDDING GENERATION COMPLETE")
    print("=" * 70)

    print()
    print("Model:")
    print("  all-MiniLM-L6-v2 (FINE-TUNED)")

    print()
    print("Embedding type:")
    print("  Fine-tuned Sentence-BERT embeddings")

    print()
    print("Device:")
    print("  CPU")

    print()
    print(f"Embedding dimension:")
    print(f"  {embedding_dimension}")

    print()
    print(f"Samples embedded:")
    print(f"  {len(df)}")

    print()
    print(f"Training samples:")
    print(f"  {len(train_idx)}")

    print()
    print(f"Testing samples:")
    print(f"  {len(test_idx)}")

    print()
    print("Saved files:")

    print(
        f"  {EMBEDDINGS_PATH}"
    )

    print(
        f"  {METADATA_PATH}"
    )

    print(
        f"  {MODEL_INFO_PATH}"
    )

    print()
    print("=" * 70)
    print("NEXT STEP")
    print("=" * 70)

    print()
    print(
        "Compare the fine-tuned embeddings against "
        "the previous pretrained SBERT embeddings."
    )

    print()
    print(
        "Do NOT overwrite the original pretrained "
        "embedding file."
    )

    print()


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print("Process interrupted by user.")

    except Exception as e:

        fail(
            "Unhandled error in main().",
            e
        )
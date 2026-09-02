"""
CareerCast - Milestone 2
Step 2: Build Sentence-BERT semantic embeddings

Pipeline:

Skills + Education + Experience + Job Description
                    ↓
             Sentence-BERT
                    ↓
          Semantic embeddings
                    ↓
      Saved career/profile embeddings
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("results/careercast_candidate_profiles.csv")

OUTPUT_DIR = Path("results/semantic_embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 32


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("CAREERCAST MILESTONE 2")
    print("STEP 2 - SENTENCE-BERT SEMANTIC EMBEDDINGS")
    print("=" * 75)

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    log("Loading candidate profiles...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {DATA_PATH.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique careers: {df['Career'].nunique()}")

    # ========================================================
    # 2. CREATE SEMANTIC TEXT
    # ========================================================

    log("Creating semantic text representations...")

    df["semantic_text"] = (
        "Skills: "
        + df["Skills"].astype(str)
        + ". Education: "
        + df["Education"].astype(str)
        + ". Experience: "
        + df["Experience"].astype(str)
        + ". Job Description: "
        + df["Job_Description"].astype(str)
    )

    # ========================================================
    # 3. LOAD SENTENCE-BERT
    # ========================================================

    log(f"Loading Sentence-BERT model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    log("Sentence-BERT model loaded successfully.")

    # ========================================================
    # 4. GENERATE EMBEDDINGS
    # ========================================================

    log("Generating semantic embeddings...")
    log("This may take some time on CPU...")

    embeddings = model.encode(
        df["semantic_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print()
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    # ========================================================
    # 5. VALIDATE EMBEDDINGS
    # ========================================================

    if len(embeddings) != len(df):
        raise ValueError(
            "Number of embeddings does not match dataset rows."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    # ========================================================
    # 6. SAVE EMBEDDINGS
    # ========================================================

    log("Saving embeddings...")

    np.save(
        OUTPUT_DIR / "profile_embeddings.npy",
        embeddings,
    )

    # ========================================================
    # 7. SAVE PROFILE INFORMATION
    # ========================================================

    profile_info = df[
        [
            "Profile_ID",
            "Career",
            "Skills",
            "Education",
            "Experience",
            "Job_Description",
        ]
    ].copy()

    profile_info.to_csv(
        OUTPUT_DIR / "profile_metadata.csv",
        index=False,
    )

    # ========================================================
    # 8. SAVE MODEL
    # ========================================================

    log("Saving Sentence-BERT model information...")

    model_info = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "number_of_profiles": int(len(embeddings)),
        "normalized": True,
        "batch_size": BATCH_SIZE,
    }

    joblib.dump(
        model_info,
        OUTPUT_DIR / "model_info.joblib",
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 75)
    print("STEP 2 COMPLETE")
    print("=" * 75)

    print(f"Profiles embedded : {len(embeddings)}")
    print(f"Embedding size    : {embeddings.shape[1]}")

    print()
    print("Saved files:")

    print(
        OUTPUT_DIR / "profile_embeddings.npy"
    )

    print(
        OUTPUT_DIR / "profile_metadata.csv"
    )

    print(
        OUTPUT_DIR / "model_info.joblib"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
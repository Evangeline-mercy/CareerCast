"""
CareerCast Milestone 2 — Step 3: Semantic Career Ranking Engine
===============================================================
Builds career-level embeddings by averaging and normalizing all profile
embeddings per career, then ranks careers for a new candidate via cosine
similarity against the 878 career embeddings.

No XGBoost retraining. No new dataset creation. Uses pre-built:
    results/semantic_embeddings/profile_embeddings.npy
    results/semantic_embeddings/profile_metadata.csv

Run:
    venv\\Scripts\\activate
    python build_career_semantic_ranker.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
import json
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMBED_DIR        = os.path.join("results", "semantic_embeddings")
PROFILE_NPY      = os.path.join(EMBED_DIR, "profile_embeddings.npy")
PROFILE_META_CSV = os.path.join(EMBED_DIR, "profile_metadata.csv")
CAREER_NPY       = os.path.join(EMBED_DIR, "career_embeddings.npy")
CAREER_LABELS    = os.path.join(EMBED_DIR, "career_labels.csv")
CAREER_META_JSON = os.path.join(EMBED_DIR, "career_embedding_metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def elapsed(start):
    return time.time() - start

# ---------------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------------

def build_candidate_text(skills, education, experience, job_description):
    """
    Construct a structured text string for the candidate.
    The Career label is intentionally excluded — only input features are used.
    """
    return (
        f"Skills: {skills} "
        f"Education: {education} "
        f"Experience: {experience} "
        f"Job Description: {job_description}"
    ).strip()

# ---------------------------------------------------------------------------
# Career embedding builder
# ---------------------------------------------------------------------------

def build_career_embeddings(profile_embeddings, metadata_df):
    """
    For each unique career, compute the mean of all its profile embeddings,
    then L2-normalize the result.

    Returns
    -------
    career_embeddings : np.ndarray, shape (n_careers, embed_dim), float32
    career_names      : list of str, length n_careers (sorted alphabetically)
    """
    careers = sorted(metadata_df["Career"].unique())
    embed_dim = profile_embeddings.shape[1]
    career_matrix = np.zeros((len(careers), embed_dim), dtype=np.float32)

    for i, career in enumerate(careers):
        mask = metadata_df["Career"].values == career
        career_profiles = profile_embeddings[mask]   # (n_profiles, embed_dim)
        career_matrix[i] = career_profiles.mean(axis=0)

    # L2-normalize each career embedding to unit length
    career_matrix = normalize(career_matrix, norm="l2").astype(np.float32)
    return career_matrix, careers

# ---------------------------------------------------------------------------
# Ranking function
# ---------------------------------------------------------------------------

def rank_careers(
    skills,
    education,
    experience,
    job_description,
    model,
    career_embeddings,
    career_names,
    top_k=10,
):
    """
    Rank careers by cosine similarity to a new candidate profile.

    Parameters
    ----------
    skills, education, experience, job_description : str
        Candidate profile fields (Career is intentionally excluded).
    model : SentenceTransformer
        Pre-loaded SBERT model.
    career_embeddings : np.ndarray, shape (n_careers, embed_dim)
        L2-normalized career embeddings.
    career_names : list of str
        Career labels in the same row-order as career_embeddings.
    top_k : int
        Number of top careers to return.

    Returns
    -------
    pd.DataFrame with columns [Rank, Career, Similarity]
    """
    # 1. Build candidate text (no Career label)
    candidate_text = build_candidate_text(
        skills, education, experience, job_description
    )

    # 2. Generate candidate embedding
    candidate_vec = model.encode(
        [candidate_text],
        convert_to_numpy=True,
        normalize_embeddings=True,   # SBERT built-in L2 normalization
        show_progress_bar=False,
    ).astype(np.float32)             # shape: (1, embed_dim)

    # 3. Cosine similarity = dot product (both sides are unit vectors)
    similarities = (career_embeddings @ candidate_vec.T).flatten()  # (n_careers,)

    # 4. Sort descending, take top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = pd.DataFrame({
        "Rank"      : range(1, top_k + 1),
        "Career"    : [career_names[i] for i in top_indices],
        "Similarity": [round(float(similarities[i]), 4) for i in top_indices],
    })
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_start = time.time()
    log("=" * 65)
    log("CareerCast Milestone 2 — Step 3: Semantic Career Ranking Engine")
    log("=" * 65)

    # -----------------------------------------------------------------------
    # Stage 1: Load profile embeddings and metadata
    # -----------------------------------------------------------------------
    log("Stage 1/6  Loading profile embeddings and metadata …")
    try:
        t = time.time()
        profile_embeddings = np.load(PROFILE_NPY).astype(np.float32)
        metadata_df = pd.read_csv(PROFILE_META_CSV)
        log(f"  Loaded in {elapsed(t):.2f}s")
        log(f"  Profile embeddings shape : {profile_embeddings.shape}")
        log(f"  Metadata rows            : {len(metadata_df)}")
        log(f"  Metadata columns         : {list(metadata_df.columns)}")
    except Exception:
        log("ERROR loading profile embeddings / metadata:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 2: Validate alignment
    # -----------------------------------------------------------------------
    log("Stage 2/6  Validating embedding ↔ metadata alignment …")
    try:
        n_embed = profile_embeddings.shape[0]
        n_meta  = len(metadata_df)
        assert n_embed == n_meta, (
            f"Mismatch: {n_embed} embeddings vs {n_meta} metadata rows"
        )

        unique_careers = sorted(metadata_df["Career"].unique())
        n_careers = len(unique_careers)
        embed_dim = profile_embeddings.shape[1]

        log(f"  OK — {n_embed} embeddings aligned with {n_meta} metadata rows")
        log(f"  Unique careers   : {n_careers}")
        log(f"  Embedding dim    : {embed_dim}")
        log(f"  Profiles / career: {n_embed / n_careers:.1f} (mean)")
    except AssertionError as e:
        log(f"VALIDATION ERROR: {e}")
        sys.exit(1)
    except Exception:
        log("ERROR during validation:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 3: Build career-level embeddings
    # -----------------------------------------------------------------------
    log("Stage 3/6  Building career-level embeddings …")
    log("           Method: mean of all profile embeddings per career → L2-normalize")
    try:
        t = time.time()
        career_embeddings, career_names = build_career_embeddings(
            profile_embeddings, metadata_df
        )
        log(f"  Completed in {elapsed(t):.2f}s")
        log(f"  Career embeddings shape  : {career_embeddings.shape}")
        log(f"  Career embeddings dtype  : {career_embeddings.dtype}")

        # Quick sanity: all rows should be unit vectors
        norms = np.linalg.norm(career_embeddings, axis=1)
        log(f"  Norm stats (expect ≈1.0) : min={norms.min():.4f}  "
            f"max={norms.max():.4f}  mean={norms.mean():.4f}")
    except Exception:
        log("ERROR building career embeddings:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 4: Save career embeddings and labels
    # -----------------------------------------------------------------------
    log("Stage 4/6  Saving career embeddings and labels …")
    try:
        os.makedirs(EMBED_DIR, exist_ok=True)

        np.save(CAREER_NPY, career_embeddings)
        log(f"  Saved → {CAREER_NPY}")

        pd.DataFrame({"Career": career_names}).to_csv(
            CAREER_LABELS, index=False
        )
        log(f"  Saved → {CAREER_LABELS}")

        metadata_out = {
            "number_of_careers" : n_careers,
            "embedding_dimension": int(embed_dim),
            "source_profiles"   : int(n_embed),
            "model_name"        : MODEL_NAME,
            "aggregation_method": "mean of all profile embeddings per career",
            "normalized"        : True,
            "normalization"     : "L2 (unit vectors)",
            "note"              : (
                "Career label was NOT included in candidate embedding input. "
                "Only Skills, Education, Experience, and Job Description are used."
            ),
        }
        with open(CAREER_META_JSON, "w") as f:
            json.dump(metadata_out, f, indent=2)
        log(f"  Saved → {CAREER_META_JSON}")
    except Exception:
        log("ERROR saving career embeddings:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 5: Load SBERT model
    # -----------------------------------------------------------------------
    log("Stage 5/6  Loading Sentence-BERT model …")
    log(f"           Model: {MODEL_NAME}")
    try:
        t = time.time()
        model = SentenceTransformer(MODEL_NAME)
        log(f"  Model loaded in {elapsed(t):.2f}s")
    except Exception:
        log("ERROR loading SentenceTransformer model:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 6: Demo — rank careers for a sample candidate
    # -----------------------------------------------------------------------
    log("Stage 6/6  Running demonstration with sample candidate …")

    # Sample candidate — realistic data science / ML profile
    sample = {
        "skills"         : "Python, SQL, Machine Learning, Pandas, NumPy",
        "education"      : "B.E. Electronics and Communication Engineering",
        "experience"     : "1 year internship in Python and data analysis",
        "job_description": (
            "Develop machine learning models, analyze data, "
            "build predictive systems and deploy data-driven applications."
        ),
    }

    print(flush=True)
    print("  ── Sample Candidate Profile ──────────────────────────", flush=True)
    print(f"  Skills      : {sample['skills']}", flush=True)
    print(f"  Education   : {sample['education']}", flush=True)
    print(f"  Experience  : {sample['experience']}", flush=True)
    print(f"  Job Desc    : {sample['job_description']}", flush=True)
    print("  ─────────────────────────────────────────────────────", flush=True)
    print(flush=True)

    try:
        t = time.time()
        top10 = rank_careers(
            skills          = sample["skills"],
            education       = sample["education"],
            experience      = sample["experience"],
            job_description = sample["job_description"],
            model           = model,
            career_embeddings = career_embeddings,
            career_names    = career_names,
            top_k           = 10,
        )
        log(f"  Ranking computed in {elapsed(t):.3f}s")

        print(flush=True)
        print("  ══════════════════════════════════════════════════", flush=True)
        print("   Top 10 Semantic Career Recommendations", flush=True)
        print("  ══════════════════════════════════════════════════", flush=True)
        header = f"  {'Rank':<6}{'Career':<45}{'Similarity':>10}"
        print(header, flush=True)
        print("  " + "-" * 62, flush=True)
        for _, row in top10.iterrows():
            print(
                f"  {int(row['Rank']):<6}{str(row['Career']):<45}{row['Similarity']:>10.4f}",
                flush=True,
            )
        print("  ══════════════════════════════════════════════════", flush=True)
        print(flush=True)

    except Exception:
        log("ERROR during ranking demonstration:")
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = elapsed(script_start)
    log("=" * 65)
    log(f"Step 3 complete!  Total time: {total:.1f}s  ({total/60:.1f} min)")
    log("Output files:")
    for f in [CAREER_NPY, CAREER_LABELS, CAREER_META_JSON]:
        size_kb = os.path.getsize(f) / 1024
        log(f"  {f}  ({size_kb:.1f} KB)")
    log("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
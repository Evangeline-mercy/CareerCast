# -*- coding: cp1252 -*-
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CAREERCAST MILESTONE 2
# SBERT JOB-DESCRIPTION-ONLY DIAGNOSTIC
# ============================================================
#
# READ-ONLY
# No model training
# No XGBoost
# No skill matching
# No hybrid scoring
# No file modification
#
# Purpose:
# Test whether Sentence-BERT alone can identify sensible
# careers for the candidate using JOB DESCRIPTION only.
# ============================================================


DATASET = "results/careercast_candidate_profiles.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


# ------------------------------------------------------------
# Candidate
# ------------------------------------------------------------

candidate_job_description = """
Develop machine learning models, analyze datasets, build
predictive systems, perform data preprocessing and
visualization, and deploy data-driven applications.
""".strip()


print("=" * 75)
print("CAREERCAST MILESTONE 2")
print("SBERT JOB-DESCRIPTION-ONLY DIAGNOSTIC")
print("=" * 75)

print()
print("READ-ONLY MODE")
print("No model training.")
print("No XGBoost.")
print("No skill matching.")
print("No hybrid ranking.")
print("No existing artifact will be modified.")

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print()
print("=" * 75)
print("1. LOADING DATASET")
print("=" * 75)

if not os.path.exists(DATASET):
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

df = pd.read_csv(DATASET)

print(f"Dataset rows       : {len(df)}")
print(f"Dataset columns    : {list(df.columns)}")

required = ["Career", "Job_Description"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

print("Required columns   : PASS")


# ------------------------------------------------------------
# Clean Job Description
# ------------------------------------------------------------

df["Job_Description"] = (
    df["Job_Description"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["Job_Description"] != ""].copy()

print(f"Usable descriptions: {len(df)}")


# ------------------------------------------------------------
# Create ONE description per career
# ------------------------------------------------------------

print()
print("=" * 75)
print("2. BUILDING CAREER-LEVEL JOB DESCRIPTIONS")
print("=" * 75)

career_df = (
    df.groupby("Career", as_index=False)
      .agg({"Job_Description": "first"})
)

print(f"Unique careers     : {len(career_df)}")


# ------------------------------------------------------------
# Load SBERT
# ------------------------------------------------------------

print()
print("=" * 75)
print("3. LOADING SENTENCE-BERT")
print("=" * 75)

print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Sentence-BERT loaded successfully.")


# ------------------------------------------------------------
# Encode candidate
# ------------------------------------------------------------

print()
print("=" * 75)
print("4. ENCODING CANDIDATE JOB DESCRIPTION")
print("=" * 75)

print()
print("Candidate Job Description:")
print("-" * 75)
print(candidate_job_description)
print("-" * 75)

candidate_embedding = model.encode(
    [candidate_job_description],
    normalize_embeddings=True,
    show_progress_bar=False
)

print(f"Candidate embedding shape: {candidate_embedding.shape}")


# ------------------------------------------------------------
# Encode career job descriptions
# ------------------------------------------------------------

print()
print("=" * 75)
print("5. ENCODING CAREER JOB DESCRIPTIONS")
print("=" * 75)

career_embeddings = model.encode(
    career_df["Job_Description"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

print(f"Career embedding shape   : {career_embeddings.shape}")


# ------------------------------------------------------------
# Cosine similarity
# ------------------------------------------------------------

print()
print("=" * 75)
print("6. CALCULATING COSINE SIMILARITY")
print("=" * 75)

similarities = cosine_similarity(
    candidate_embedding,
    career_embeddings
)[0]

career_df["Similarity"] = similarities

print(f"Similarity count : {len(similarities)}")
print(f"Minimum          : {similarities.min():.6f}")
print(f"Maximum          : {similarities.max():.6f}")
print(f"Mean             : {similarities.mean():.6f}")
print(f"Median           : {np.median(similarities):.6f}")


# ------------------------------------------------------------
# Top 15
# ------------------------------------------------------------

print()
print("=" * 75)
print("7. TOP 15 CAREERS — SBERT JOB DESCRIPTION ONLY")
print("=" * 75)

top15 = (
    career_df
    .sort_values("Similarity", ascending=False)
    .head(15)
    .reset_index(drop=True)
)

print()
print(f"{'Rank':<6}{'Career':<60}{'Similarity':>12}")
print("-" * 80)

for i, row in top15.iterrows():
    career = str(row["Career"])
    similarity = float(row["Similarity"])

    print(
        f"{i + 1:<6}"
        f"{career[:58]:<60}"
        f"{similarity:>12.6f}"
    )


# ------------------------------------------------------------
# Important expected careers
# ------------------------------------------------------------

expected_careers = [
    "Software Developers",
    "Web Developers",
    "Computer Systems Analysts",
    "Data Warehousing Specialists",
    "Database Architects",
    "Database Administrators",
    "Computer and Information Research Scientists",
    "Computer Network Architects",
    "Network and Computer Systems Administrators",
    "Telecommunications Engineering Specialists",
    "Sales Engineers",
]


print()
print("=" * 75)
print("8. EXPECTED AI / DATA / SOFTWARE CAREER CHECK")
print("=" * 75)

lookup = career_df.set_index("Career")["Similarity"]

print()
print(f"{'Career':<60}{'Similarity':>12}")
print("-" * 75)

for career in expected_careers:

    if career in lookup.index:
        print(
            f"{career:<60}"
            f"{lookup[career]:>12.6f}"
        )
    else:
        print(
            f"{career:<60}"
            f"{'NOT FOUND':>12}"
        )


# ------------------------------------------------------------
# Top 15 descriptions
# ------------------------------------------------------------

print()
print("=" * 75)
print("9. TOP 15 JOB DESCRIPTION CONTENT")
print("=" * 75)

for i, row in top15.iterrows():

    print()
    print(f"{i + 1}. {row['Career']}")
    print(f"Similarity: {row['Similarity']:.6f}")
    print("Job Description:")
    print(str(row["Job_Description"])[:700])
    print("-" * 75)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print()
print("=" * 75)
print("DIAGNOSTIC COMPLETE")
print("=" * 75)

print()
print("IMPORTANT:")
print("- This test used ONLY candidate Job Description.")
print("- Skills were NOT used.")
print("- Education was NOT used.")
print("- Experience was NOT used.")
print("- RIASEC was NOT used.")
print("- XGBoost was NOT used.")
print("- Hybrid ranking was NOT used.")
print("- No existing artifact was modified.")

print()
print("Use the TOP 15 results to determine whether")
print("SBERT semantic similarity is behaving sensibly.")
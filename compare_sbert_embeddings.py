# ============================================================
# CAREERCAST MILESTONE 2
# TASK 2 - OLD vs FINE-TUNED SBERT COMPARISON
# ============================================================

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "results/careercast_candidate_profiles.csv"

OLD_EMBEDDINGS_PATH = (
    "results/semantic_embeddings/career_embeddings.npy"
)

OLD_LABELS_PATH = (
    "results/semantic_embeddings/career_labels.csv"
)

NEW_EMBEDDINGS_PATH = (
    "results/semantic_embeddings/"
    "finetuned_career_embeddings.npy"
)

NEW_LABELS_PATH = (
    "results/semantic_embeddings/"
    "finetuned_career_labels.csv"
)

FINETUNED_MODEL_PATH = (
    "results/semantic_embeddings/sbert_finetuned"
)


# ============================================================
# CANDIDATE
# ============================================================

CANDIDATE = {
    "skills": (
        "Python, SQL, Machine Learning, Pandas, "
        "NumPy, Data Analysis, TensorFlow"
    ),

    "education": (
        "B.E. Electronics and Communication Engineering"
    ),

    "experience": (
        "6 months internship experience in Python, "
        "data analysis and machine learning projects"
    ),

    "job_description": (
        "Develop machine learning models, analyze datasets, "
        "build predictive systems, perform data preprocessing "
        "and visualization, and deploy data-driven applications."
    )
}


# ============================================================
# BUILD CANDIDATE TEXT
# ============================================================

candidate_text = (
    f"Skills: {CANDIDATE['skills']}. "
    f"Education: {CANDIDATE['education']}. "
    f"Experience: {CANDIDATE['experience']}. "
    f"Job Description: {CANDIDATE['job_description']}"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 75)
print("CAREERCAST MILESTONE 2")
print("TASK 2 - OLD vs FINE-TUNED SBERT")
print("=" * 75)

print("\nLoading old embeddings...")

old_embeddings = np.load(
    OLD_EMBEDDINGS_PATH
)

old_labels_df = pd.read_csv(
    OLD_LABELS_PATH
)

old_labels = (
    old_labels_df["Career"]
    .astype(str)
    .tolist()
)

print(
    "Old embedding shape:",
    old_embeddings.shape
)

print(
    "Old career count:",
    len(old_labels)
)


print("\nLoading fine-tuned embeddings...")

new_embeddings = np.load(
    NEW_EMBEDDINGS_PATH
)

new_labels_df = pd.read_csv(
    NEW_LABELS_PATH
)

new_labels = (
    new_labels_df["Career"]
    .astype(str)
    .tolist()
)

print(
    "Fine-tuned embedding shape:",
    new_embeddings.shape
)

print(
    "Fine-tuned career count:",
    len(new_labels)
)


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

if old_embeddings.shape != new_embeddings.shape:

    raise ValueError(
        "Old and fine-tuned embedding shapes do not match."
    )

if old_labels != new_labels:

    raise ValueError(
        "Career label order does not match between "
        "old and fine-tuned embeddings."
    )

print("\nCareer alignment: PASS")


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================

print("\nLoading fine-tuned SBERT model...")

model = SentenceTransformer(
    FINETUNED_MODEL_PATH
)

print(
    "Fine-tuned model loaded successfully."
)


# ============================================================
# GENERATE CANDIDATE EMBEDDINGS
# ============================================================

print("\nGenerating candidate embedding...")

new_candidate_embedding = model.encode(
    [candidate_text],
    normalize_embeddings=True,
    convert_to_numpy=True
)[0]


# ============================================================
# NORMALIZE OLD EMBEDDINGS
# ============================================================

old_norms = np.linalg.norm(
    old_embeddings,
    axis=1,
    keepdims=True
)

old_normalized = (
    old_embeddings /
    (old_norms + 1e-12)
)


# ============================================================
# NORMALIZE NEW EMBEDDINGS
# ============================================================

new_norms = np.linalg.norm(
    new_embeddings,
    axis=1,
    keepdims=True
)

new_normalized = (
    new_embeddings /
    (new_norms + 1e-12)
)


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

old_candidate_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

old_candidate_embedding = old_candidate_model.encode(
    [candidate_text],
    normalize_embeddings=True,
    convert_to_numpy=True
)[0]


old_scores = (
    old_normalized @ old_candidate_embedding
)

new_scores = (
    new_normalized @ new_candidate_embedding
)


# ============================================================
# TOP 15
# ============================================================

TOP_K = 15

old_indices = np.argsort(
    old_scores
)[::-1][:TOP_K]

new_indices = np.argsort(
    new_scores
)[::-1][:TOP_K]


# ============================================================
# DISPLAY OLD RESULTS
# ============================================================

print("\n" + "=" * 75)
print("TOP 15 - ORIGINAL SBERT")
print("=" * 75)

for rank, index in enumerate(
    old_indices,
    start=1
):

    print(
        f"{rank:2}. "
        f"{old_labels[index]} "
        f"-> {old_scores[index]:.4f}"
    )


# ============================================================
# DISPLAY NEW RESULTS
# ============================================================

print("\n" + "=" * 75)
print("TOP 15 - FINE-TUNED SBERT")
print("=" * 75)

for rank, index in enumerate(
    new_indices,
    start=1
):

    print(
        f"{rank:2}. "
        f"{new_labels[index]} "
        f"-> {new_scores[index]:.4f}"
    )


# ============================================================
# OVERLAP
# ============================================================

old_top = {
    old_labels[index]
    for index in old_indices
}

new_top = {
    new_labels[index]
    for index in new_indices
}

overlap = old_top.intersection(
    new_top
)


# ============================================================
# STATISTICS
# ============================================================

old_top_score = float(
    old_scores[old_indices[0]]
)

new_top_score = float(
    new_scores[new_indices[0]]
)

print("\n" + "=" * 75)
print("COMPARISON SUMMARY")
print("=" * 75)

print(
    "Original SBERT top score :",
    round(old_top_score, 4)
)

print(
    "Fine-tuned SBERT top score:",
    round(new_top_score, 4)
)

print(
    "Top-15 career overlap     :",
    len(overlap),
    "/ 15"
)

print(
    "Score change              :",
    round(
        new_top_score - old_top_score,
        4
    )
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_rows = []

for rank in range(TOP_K):

    old_index = old_indices[rank]
    new_index = new_indices[rank]

    comparison_rows.append(
        {
            "rank": rank + 1,

            "original_career":
                old_labels[old_index],

            "original_score":
                round(
                    float(old_scores[old_index]),
                    6
                ),

            "finetuned_career":
                new_labels[new_index],

            "finetuned_score":
                round(
                    float(new_scores[new_index]),
                    6
                )
        }
    )

comparison_df = pd.DataFrame(
    comparison_rows
)

output_path = (
    "results/semantic_embeddings/"
    "sbert_old_vs_finetuned_comparison.csv"
)

comparison_df.to_csv(
    output_path,
    index=False
)

print(
    "\nComparison saved to:"
)

print(
    output_path
)

print("\n" + "=" * 75)
print("TASK 2 VALIDATION COMPLETE")
print("=" * 75)
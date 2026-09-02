import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATASET = "results/milestone2_training/career_profile_training_dataset.csv"

print("=" * 90)
print("CAREERCAST - SBERT EMBEDDING SIMILARITY TEST")
print("=" * 90)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATASET)
df["skills"] = df["skills"].fillna("")

# ------------------------------------------------------------
# Load SBERT
# ------------------------------------------------------------

print("\nLoading SBERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------------------
# Controlled test profiles
# ------------------------------------------------------------

tests = {
    "C003_VLSI":
        "Verilog, VLSI, RTL Design, Digital Electronics, FPGA",

    "C004_EMBEDDED":
        "Arduino, Embedded C, Microcontrollers, IoT, Embedded Systems",

    "C005_WEB":
        "HTML, CSS, JavaScript, React, Web Development",

    "C006_DATABASE":
        "SQL, Database Management, Database Design, Python",

    "C009_SIGNAL":
        "MATLAB, Signal Processing, Digital Signal Processing, Communication Systems",

    "C010_CLOUD":
        "AWS, Linux, Cloud Computing, Python, Networking",
}

# ------------------------------------------------------------
# Create one representative profile per career
# ------------------------------------------------------------

careers = [
    "VLSI Engineer",
    "IoT Engineer",
    "React Developer",
    "Database Administrator",
    "Signal Processing Engineer",
    "AWS Solutions Architect",
]

career_profiles = []

for career in careers:

    samples = df[df["career"] == career]["skills"].head(20).tolist()

    # Combine several genuine training samples
    representative = " | ".join(samples)

    career_profiles.append(representative)

# ------------------------------------------------------------
# Generate embeddings
# ------------------------------------------------------------

print("\nGenerating embeddings...")

career_embeddings = model.encode(
    career_profiles,
    normalize_embeddings=True,
    show_progress_bar=True,
)

test_names = list(tests.keys())
test_texts = list(tests.values())

test_embeddings = model.encode(
    test_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# ------------------------------------------------------------
# Similarity
# ------------------------------------------------------------

similarities = cosine_similarity(
    test_embeddings,
    career_embeddings,
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

for i, test_name in enumerate(test_names):

    print("\n" + "=" * 90)
    print(test_name)
    print("-" * 90)
    print("Input:", tests[test_name])

    scores = similarities[i]

    ranking = np.argsort(scores)[::-1]

    print("\nSBERT similarity ranking:")

    for rank, idx in enumerate(ranking, start=1):

        print(
            f"{rank}. "
            f"{careers[idx]:<35} "
            f"{scores[idx]:.4f}"
        )

print("\n" + "=" * 90)
print("TEST COMPLETED")
print("=" * 90)
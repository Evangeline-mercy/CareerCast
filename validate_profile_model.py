import pandas as pd
import joblib
import os
import re

MODEL_DIR = "results/milestone2_profile_model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

VALIDATION_FILE = "careercast_curated_validation.csv"

OUTPUT_FILE = os.path.join(
    MODEL_DIR,
    "curated_validation_predictions.csv"
)


# ============================================================
# CAREER NAME NORMALIZATION
# ============================================================

def normalize_career_name(name):
    """
    Converts validation career names into the same naming
    convention used by the trained model.
    """

    name = str(name).strip()

    career_mapping = {
        "Data Scientists": "Data Scientist",
        "Data Analysts": "Data Analyst",
        "Machine Learning Engineers": "Machine Learning Engineer",
        "Frontend Developers": "Frontend Developer",
        "Embedded Systems Engineers": "Embedded Systems Engineer",
        "IoT Engineers": "IoT Engineer",
        "Electronics Engineers": "Electronics Engineer",
        "Electrical Engineers": "Electrical Engineer",
        "Firmware Engineers": "Firmware Engineer",
        "VLSI Engineers": "VLSI Engineer",
        "Signal Processing Engineers": "Signal Processing Engineer",
        "Cloud Engineers": "Cloud Engineer",
        "DevOps Engineers": "DevOps Engineer",
        "Software Developers": "Software Developer",
        "Web Developers": "Frontend Developer",
        "Computer Hardware Engineers": "Computer Hardware Engineer",
        "Computer and Information Research Scientists":
            "Computer and Information Research Scientist",
        "Operations Research Analysts":
            "Operations Research Analyst",
        "Statisticians": "R Developer",
    }

    return career_mapping.get(name, name)


# ============================================================
# START
# ============================================================

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("EXTERNAL VALIDATION - PROFILE LOGISTIC REGRESSION")
print("=" * 90)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\n[1] Loading trained model...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Model loaded successfully.")
print("Vectorizer loaded successfully.")


# ============================================================
# 2. LOAD VALIDATION DATASET
# ============================================================

print("\n[2] Loading curated validation dataset...")

df = pd.read_csv(VALIDATION_FILE)

print("Validation rows :", len(df))
print("Validation columns:", list(df.columns))


# ============================================================
# 3. PREPARE SKILL PROFILES
# ============================================================

print("\n[3] Preparing candidate skill profiles...")

df["skills"] = (
    df["skills"]
    .fillna("")
    .astype(str)
    .str.lower()
)

X_validation = vectorizer.transform(df["skills"])

print("Validation TF-IDF shape:", X_validation.shape)


# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

print("\n[4] Generating predictions...")

probabilities = model.predict_proba(X_validation)

classes = model.classes_


# ============================================================
# 5. TOP-3 PREDICTIONS
# ============================================================

results = []

top1_correct = 0
top3_correct = 0

print("\n")
print("=" * 90)
print("CANDIDATE-WISE TOP-3 CAREER PREDICTIONS")
print("=" * 90)


for i, row in df.iterrows():

    candidate_id = row["candidate_id"]
    skills = row["skills"]
    expected_raw = row["expected_careers"]

    expected_careers = [
        normalize_career_name(x)
        for x in str(expected_raw).split("|")
        if str(x).strip()
    ]

    # Sort probabilities from highest to lowest
    ranked_indices = probabilities[i].argsort()[::-1][:3]

    top3 = []

    for rank, index in enumerate(ranked_indices, start=1):

        career = classes[index]
        probability = probabilities[i][index]

        top3.append(career)

        if rank == 1:
            top1_career = career
            top1_probability = probability

        if rank == 2:
            top2_career = career
            top2_probability = probability

        if rank == 3:
            top3_career = career
            top3_probability = probability


    # ========================================================
    # MATCH CHECK
    # ========================================================

    top1_match = top1_career in expected_careers
    top3_match = any(
        career in expected_careers
        for career in top3
    )

    if top1_match:
        top1_correct += 1

    if top3_match:
        top3_correct += 1


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n" + "-" * 90)

    print("Candidate :", candidate_id)

    print("Skills    :", row["skills"])

    print("Expected  :", expected_raw)

    print("\nNormalized Expected Careers:")

    for career in expected_careers:
        print("  -", career)

    print("\nTop-1:")
    print(
        f"  {top1_career} "
        f"({top1_probability * 100:.2f}%)"
    )

    print("Top-2:")
    print(
        f"  {top2_career} "
        f"({top2_probability * 100:.2f}%)"
    )

    print("Top-3:")
    print(
        f"  {top3_career} "
        f"({top3_probability * 100:.2f}%)"
    )

    print("\nTop-1 Match :", top1_match)
    print("Top-3 Match :", top3_match)


    # ========================================================
    # STORE RESULT
    # ========================================================

    results.append({
        "candidate_id": candidate_id,
        "skills": row["skills"],
        "expected_careers": expected_raw,

        "normalized_expected_careers":
            "|".join(expected_careers),

        "top1_career": top1_career,
        "top1_probability": round(
            top1_probability,
            6
        ),

        "top2_career": top2_career,
        "top2_probability": round(
            top2_probability,
            6
        ),

        "top3_career": top3_career,
        "top3_probability": round(
            top3_probability,
            6
        ),

        "top1_match": top1_match,
        "top3_match": top3_match
    })


# ============================================================
# 6. FINAL VALIDATION RESULTS
# ============================================================

total_candidates = len(df)

top1_accuracy = (
    top1_correct / total_candidates
    if total_candidates > 0
    else 0
)

top3_accuracy = (
    top3_correct / total_candidates
    if total_candidates > 0
    else 0
)


print("\n")
print("=" * 90)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 90)

print("Candidates tested :", total_candidates)

print(
    f"Top-1 accuracy    : "
    f"{top1_accuracy:.4f} "
    f"({top1_accuracy * 100:.2f}%)"
)

print(
    f"Top-3 accuracy    : "
    f"{top3_accuracy:.4f} "
    f"({top3_accuracy * 100:.2f}%)"
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(os.path.abspath(OUTPUT_FILE))

print("\n" + "=" * 90)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 90)
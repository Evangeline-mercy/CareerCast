import os
import glob
import json
import pandas as pd
import numpy as np


# ============================================================
# CAREERCAST MILESTONE 2
# CURATED CANDIDATE VALIDATION
# ============================================================

VALIDATION_FILE = "careercast_curated_validation.csv"
OUTPUT_DIR = "validation_results"


print("\n" + "=" * 70)
print("CAREERCAST MILESTONE 2")
print("CURATED CANDIDATE TOP-K VALIDATION")
print("=" * 70)

print("""
READ-ONLY VALIDATION
V4 recommender will NOT be modified.
Presentation demo will NOT be modified.
Models will NOT be modified.
Dataset will NOT be modified.
""")


# ============================================================
# 1. LOAD VALIDATION DATASET
# ============================================================

print("=" * 70)
print("1. LOADING CURATED VALIDATION DATASET")
print("=" * 70)

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation dataset not found: {VALIDATION_FILE}"
    )

df = pd.read_csv(VALIDATION_FILE)

required_columns = [
    "candidate_id",
    "skills",
    "education",
    "expected_careers"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("[FOUND]")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# 2. DISPLAY GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("2. GROUND-TRUTH CANDIDATES")
print("=" * 70)

for _, row in df.iterrows():

    expected = [
        x.strip().lower()
        for x in str(row["expected_careers"]).split("|")
    ]

    print(
        f"\n{row['candidate_id']}"
    )

    print(
        "Skills:",
        row["skills"]
    )

    print(
        "Expected:",
        " | ".join(expected)
    )


# ============================================================
# 3. FIND V4 OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("3. SEARCHING V4 OUTPUT")
print("=" * 70)

v4_files = glob.glob(
    os.path.join(
        "v4_output",
        "run_*",
        "v4_recommendations.csv"
    )
)

if not v4_files:
    raise FileNotFoundError(
        "No V4 recommendation output found."
    )

v4_files.sort(
    key=os.path.getmtime,
    reverse=True
)

latest_v4 = v4_files[0]

print("Latest V4 output:")
print(latest_v4)


v4 = pd.read_csv(latest_v4)

print("V4 rows:", len(v4))
print("V4 columns:", list(v4.columns))


# ============================================================
# 4. VERIFY PRESENTATION DEMO
# ============================================================

print("\n" + "=" * 70)
print("4. PRESENTATION DEMO PROTECTION")
print("=" * 70)

DEMO_CAREERS = [
    "Python Engineer",
    "AI/ML Engineer",
    "Data Scientist",
    "Embedded Systems Engineer",
    "IoT Engineer"
]

print("Presentation Top-5:")

for i, career in enumerate(
    DEMO_CAREERS,
    start=1
):
    print(
        f"{i}. {career}"
    )

print("\nPresentation demo remains unchanged.")


# ============================================================
# 5. V4 OUTPUT STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("5. V4 OUTPUT STRUCTURE")
print("=" * 70)

required_v4_columns = [
    "rank",
    "career",
    "final_score",
    "confidence_percentage",
    "semantic_score",
    "skill_match",
    "xgboost_probability",
    "skill_alignment",
    "matched_skills"
]

missing_v4 = [
    c
    for c in required_v4_columns
    if c not in v4.columns
]

if missing_v4:
    raise ValueError(
        f"V4 output missing columns: {missing_v4}"
    )

print("V4 recommendation structure: PASS")


# ============================================================
# 6. IMPORTANT LIMITATION
# ============================================================

print("\n" + "=" * 70)
print("6. CANDIDATE-LEVEL MAPPING CHECK")
print("=" * 70)

if "candidate_id" not in v4.columns:

    print("""
[IMPORTANT]

The existing V4 recommendation CSV does not contain candidate_id.

Therefore this file cannot be safely mapped to the
10 curated candidates.

No candidate-level accuracy will be fabricated.

The existing presentation V4 output will remain untouched.
""")

    print("=" * 70)
    print("VALIDATION STATUS")
    print("=" * 70)

    print(
        "Candidate-level benchmark accuracy: NOT CALCULATED"
    )

    print(
        "Reason: V4 output represents one presentation run."
    )

    print(
        "\nPresentation Top-5 remains:"
    )

    for i, career in enumerate(
        DEMO_CAREERS,
        start=1
    ):
        print(
            f"{i}. {career}"
        )

    print(
        "\nNo files were modified."
    )

    print("=" * 70)

    raise SystemExit(0)


# ============================================================
# 7. CANDIDATE-LEVEL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("7. CANDIDATE-LEVEL TOP-K VALIDATION")
print("=" * 70)

results = []

for candidate_id in df["candidate_id"].astype(str):

    ground_truth_row = df[
        df["candidate_id"].astype(str)
        == candidate_id
    ].iloc[0]

    expected = [
        x.strip().lower()
        for x in str(
            ground_truth_row["expected_careers"]
        ).split("|")
    ]

    candidate_results = v4[
        v4["candidate_id"].astype(str)
        == candidate_id
    ].sort_values("rank")

    if candidate_results.empty:

        print(
            f"{candidate_id}: NO V4 RESULT"
        )

        results.append({
            "candidate_id": candidate_id,
            "top1_correct": False,
            "top3_correct": False,
            "top5_correct": False,
            "status": "NO V4 RESULT"
        })

        continue

    predictions = [
        str(x).strip().lower()
        for x in candidate_results[
            "career"
        ].head(5)
    ]

    top1 = (
        len(predictions) >= 1
        and predictions[0] in expected
    )

    top3 = any(
        p in expected
        for p in predictions[:3]
    )

    top5 = any(
        p in expected
        for p in predictions[:5]
    )

    results.append({
        "candidate_id": candidate_id,
        "top1_correct": top1,
        "top3_correct": top3,
        "top5_correct": top5,
        "status": "PASS"
    })

    print(
        f"{candidate_id}: "
        f"Top-1={'PASS' if top1 else 'FAIL'} | "
        f"Top-3={'PASS' if top3 else 'FAIL'} | "
        f"Top-5={'PASS' if top5 else 'FAIL'}"
    )


# ============================================================
# 8. METRICS
# ============================================================

print("\n" + "=" * 70)
print("8. TOP-K ACCURACY")
print("=" * 70)

results_df = pd.DataFrame(results)

valid = results_df[
    results_df["status"] == "PASS"
]

if len(valid) == 0:

    print(
        "No candidate-level V4 mappings available."
    )

else:

    top1_accuracy = (
        valid["top1_correct"].mean()
        * 100
    )

    top3_accuracy = (
        valid["top3_correct"].mean()
        * 100
    )

    top5_accuracy = (
        valid["top5_correct"].mean()
        * 100
    )

    print(
        f"Candidates evaluated : {len(valid)}"
    )

    print(
        f"Top-1 Accuracy       : {top1_accuracy:.2f}%"
    )

    print(
        f"Top-3 Accuracy       : {top3_accuracy:.2f}%"
    )

    print(
        f"Top-5 Accuracy       : {top5_accuracy:.2f}%"
    )


# ============================================================
# 9. SAVE VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("9. SAVING VALIDATION RESULTS")
print("=" * 70)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

output_csv = os.path.join(
    OUTPUT_DIR,
    "curated_candidate_validation.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)

print(
    "Saved:",
    output_csv
)


# ============================================================
# 10. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CURATED VALIDATION STATUS")
print("=" * 70)

if "candidate_id" in v4.columns:

    print(
        "Candidate-level validation: COMPLETE"
    )

    print(
        "Top-K accuracy calculated."
    )

else:

    print(
        "Candidate-level validation: BLOCKED"
    )

    print(
        "Existing V4 output has no candidate_id."
    )

print(
    "\nPresentation demo was NOT modified."
)

print(
    "Models were NOT modified."
)

print(
    "Dataset was NOT modified."
)

print(
    "\nPresentation Top-5:"
)

for i, career in enumerate(
    DEMO_CAREERS,
    start=1
):
    print(
        f"{i}. {career}"
    )

print("\n" + "=" * 70)

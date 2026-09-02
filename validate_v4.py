# ============================================================
# validate_v4.py
# CAREERCAST V4 - DEMO VALIDATION ONLY
# ============================================================

import os
import glob
import ast
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

V4_OUTPUT_ROOT = "v4_output"

TOP_K = 5

DEMO_CAREERS = [
    "Python Engineer",
    "AI/ML Engineer",
    "Data Scientist",
    "Embedded Systems Engineer",
    "IoT Engineer"
]


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 - CAREER RECOMMENDER VALIDATION")
print("=" * 70)


# ============================================================
# FIND LATEST V4 OUTPUT
# ============================================================

print("\nSearching latest V4 recommendation output...")


if not os.path.exists(V4_OUTPUT_ROOT):
    raise FileNotFoundError(
        f"V4 output folder not found: {V4_OUTPUT_ROOT}"
    )


run_directories = [
    path
    for path in glob.glob(
        os.path.join(
            V4_OUTPUT_ROOT,
            "run_*"
        )
    )
    if os.path.isdir(path)
]


if not run_directories:
    raise FileNotFoundError(
        "No V4 run directories found."
    )


latest_run = max(
    run_directories,
    key=os.path.getmtime
)


recommendation_file = os.path.join(
    latest_run,
    "v4_recommendations.csv"
)


if not os.path.exists(recommendation_file):
    raise FileNotFoundError(
        "v4_recommendations.csv not found:\n"
        + recommendation_file
    )


recommendations = pd.read_csv(
    recommendation_file
)


print(
    "Latest V4 run:",
    latest_run
)

print(
    "Recommendation rows:",
    len(recommendations)
)

print(
    "Columns:",
    list(recommendations.columns)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "rank",
    "career",
    "final_score",
    "confidence_percentage",
    "semantic_score",
    "semantic_normalized",
    "skill_match",
    "skill_match_normalized",
    "xgboost_probability",
    "xgboost_normalized",
    "matched_skills",
    "skill_alignment"
]


missing = [
    column
    for column in required_columns
    if column not in recommendations.columns
]


if missing:
    raise ValueError(
        "Missing required columns: "
        + str(missing)
    )


print(
    "\nV4 recommendation structure: PASS"
)


# ============================================================
# SORT BY RANK
# ============================================================

recommendations = (
    recommendations
    .sort_values(
        "rank"
    )
    .reset_index(
        drop=True
    )
)


top5 = recommendations.head(
    TOP_K
).copy()


actual_careers = (
    top5["career"]
    .astype(str)
    .tolist()
)


# ============================================================
# EXPECTED DEMO
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED PRESENTATION DEMO")
print("=" * 70)


for rank, career in enumerate(
    DEMO_CAREERS,
    start=1
):
    print(
        f"{rank}. {career}"
    )


# ============================================================
# ACTUAL V4 OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL V4 TOP-5")
print("=" * 70)


for rank, career in enumerate(
    actual_careers,
    start=1
):
    print(
        f"{rank}. {career}"
    )


# ============================================================
# DEMO CAREER CHECK
# ============================================================

print("\n" + "=" * 70)
print("DEMO CAREER VALIDATION")
print("=" * 70)


demo_pass = True


for rank in range(TOP_K):

    expected = DEMO_CAREERS[rank]

    actual = actual_careers[rank]

    if expected == actual:

        print(
            f"Rank {rank + 1}: PASS"
        )

        print(
            f"   Expected: {expected}"
        )

        print(
            f"   Actual  : {actual}"
        )

    else:

        print(
            f"Rank {rank + 1}: FAIL"
        )

        print(
            f"   Expected: {expected}"
        )

        print(
            f"   Actual  : {actual}"
        )

        demo_pass = False


# ============================================================
# DISPLAY TOP-5 DETAILS
# ============================================================

print("\n" + "=" * 70)
print("TOP-5 DEMO DETAILS")
print("=" * 70)


display_columns = [
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


print(
    top5[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SCORE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SCORE VALIDATION")
print("=" * 70)


score_columns = [
    "final_score",
    "confidence_percentage",
    "semantic_score",
    "semantic_normalized",
    "skill_match",
    "skill_match_normalized",
    "xgboost_probability",
    "xgboost_normalized",
    "skill_alignment"
]


score_pass = True


for column in score_columns:

    values = pd.to_numeric(
        top5[column],
        errors="coerce"
    )

    if values.isna().any():

        print(
            f"{column}: FAIL"
        )

        score_pass = False

    else:

        print(
            f"{column}: PASS"
        )


# ============================================================
# RANK CHECK
# ============================================================

print("\n" + "=" * 70)
print("RANK VALIDATION")
print("=" * 70)


expected_ranks = [
    1,
    2,
    3,
    4,
    5
]


actual_ranks = (
    top5["rank"]
    .astype(int)
    .tolist()
)


if actual_ranks == expected_ranks:

    print(
        "Rank sequence: PASS"
    )

    rank_pass = True

else:

    print(
        "Rank sequence: FAIL"
    )

    print(
        "Expected:",
        expected_ranks
    )

    print(
        "Actual:",
        actual_ranks
    )

    rank_pass = False


# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CAREER CHECK")
print("=" * 70)


if top5["career"].duplicated().any():

    print(
        "Duplicate careers: FAIL"
    )

    duplicate_pass = False

else:

    print(
        "Duplicate careers: NONE"
    )

    duplicate_pass = True


# ============================================================
# MATCHED SKILL CHECK
# ============================================================

print("\n" + "=" * 70)
print("MATCHED SKILLS CHECK")
print("=" * 70)


matched_skill_pass = True


for _, row in top5.iterrows():

    career = row["career"]

    try:

        value = row["matched_skills"]

        if isinstance(
            value,
            str
        ):

            skills = ast.literal_eval(
                value
            )

        else:

            skills = value

        if skills is None:
            skills = []

        if len(skills) != len(
            set(skills)
        ):

            print(
                f"{career}: FAIL - duplicate skills"
            )

            matched_skill_pass = False

        else:

            print(
                f"{career}: PASS"
            )

    except Exception:

        print(
            f"{career}: FAIL - invalid matched_skills"
        )

        matched_skill_pass = False


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION STATUS")
print("=" * 70)


if (
    demo_pass
    and score_pass
    and rank_pass
    and duplicate_pass
    and matched_skill_pass
):

    print(
        "\nV4 DEMO VALIDATION: PASS"
    )

    print(
        "\nPresentation Top-5:"
    )

    for rank, career in enumerate(
        DEMO_CAREERS,
        start=1
    ):

        print(
            f"{rank}. {career}"
        )

else:

    print(
        "\nV4 DEMO VALIDATION: FAIL"
    )


print(
    "\nV4 recommender was NOT modified."
)

print(
    "Models were NOT modified."
)

print(
    "Dataset was NOT modified."
)

print(
    "\nValidated file:"
)

print(
    recommendation_file
)
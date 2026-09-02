import os
import json
import glob
import ast
import numpy as np
import pandas as pd

print("\n" + "=" * 70)
print("CAREERCAST MILESTONE 2")
print("BENCHMARK / CURATED VALIDATION")
print("=" * 70)

print("\nREAD-ONLY MODE")
print("V4 recommender will NOT be modified.")
print("Presentation demo will NOT be modified.")
print("Models will NOT be modified.")
print("No fake benchmark data will be created.")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. FIND CURATED VALIDATION DATASET
# ============================================================

print("\n" + "=" * 70)
print("1. SEARCHING VALIDATION DATASET")
print("=" * 70)

candidates = [
    os.path.join(
        PROJECT_ROOT,
        "careercast_curated_validation.csv"
    ),
    os.path.join(
        PROJECT_ROOT,
        "validation",
        "careercast_curated_validation.csv"
    )
]

validation_path = None

for path in candidates:
    if os.path.isfile(path):
        validation_path = path
        break

if validation_path is None:
    raise FileNotFoundError(
        "careercast_curated_validation.csv not found."
    )

print("[FOUND]")
print(validation_path)

df = pd.read_csv(validation_path)

print("\nRows:", len(df))
print("Columns:")
print(list(df.columns))

# ============================================================
# 2. INSPECT DATASET
# ============================================================

print("\n" + "=" * 70)
print("2. DATASET INSPECTION")
print("=" * 70)

for column in df.columns:
    print(
        f"{column}: "
        f"dtype={df[column].dtype}, "
        f"non-null={df[column].notna().sum()}"
    )

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

# ============================================================
# 3. FIND CAREER / EXPECTED CAREER COLUMN
# ============================================================

career_candidates = [
    "expected_career",
    "expected_careers",
    "career",
    "target_career",
    "target",
    "label",
    "Career",
    "Expected Career",
    "Expected_Career"
]

career_column = None

for column in career_candidates:
    if column in df.columns:
        career_column = column
        break

if career_column is None:

    possible = [
        column
        for column in df.columns
        if "career" in column.lower()
        or "target" in column.lower()
        or "label" in column.lower()
    ]

    if len(possible) == 1:
        career_column = possible[0]

if career_column is None:

    print("\n[WARNING]")
    print(
        "No ground-truth career column could be "
        "identified automatically."
    )

    print(
        "\nThis dataset cannot be used for "
        "accuracy/Top-K validation yet."
    )

    print(
        "\nAvailable columns:"
    )

    for column in df.columns:
        print(" -", column)

    print(
        "\nNo fake benchmark score will be generated."
    )

    raise SystemExit(0)

print("\nGround-truth career column:")
print(career_column)

# ============================================================
# 4. FIND V4 OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("3. LOADING V4 RECOMMENDATION OUTPUT")
print("=" * 70)

pattern = os.path.join(
    PROJECT_ROOT,
    "v4_output",
    "run_*",
    "v4_recommendations.csv"
)

v4_files = glob.glob(pattern)

if not v4_files:
    raise FileNotFoundError(
        "No V4 recommendation output found."
    )

v4_path = max(
    v4_files,
    key=os.path.getmtime
)

print("Latest V4 output:")
print(v4_path)

v4 = pd.read_csv(v4_path)

print("\nV4 rows:", len(v4))
print("V4 columns:")
print(list(v4.columns))

required_columns = [
    "rank",
    "career",
    "final_score"
]

missing = [
    column
    for column in required_columns
    if column not in v4.columns
]

if missing:
    raise ValueError(
        f"V4 output missing columns: {missing}"
    )

# ============================================================
# 5. PRESENTATION DEMO
# ============================================================

print("\n" + "=" * 70)
print("4. PRESENTATION DEMO")
print("=" * 70)

DEMO_CAREERS = [
    "Python Engineer",
    "AI/ML Engineer",
    "Data Scientist",
    "Embedded Systems Engineer",
    "IoT Engineer"
]

for i, career in enumerate(
    DEMO_CAREERS,
    start=1
):
    print(f"{i}. {career}")

print(
    "\nPresentation demo remains unchanged."
)

# ============================================================
# 6. DETERMINE WHETHER DATASET CAN BE EVALUATED
# ============================================================

print("\n" + "=" * 70)
print("5. GROUND-TRUTH VALIDATION")
print("=" * 70)

expected_values = (
    df[career_column]
    .dropna()
    .astype(str)
    .str.strip()
)

print(
    "Ground-truth career values:",
    len(expected_values)
)

print("\nUnique ground-truth careers:")

for career in expected_values.unique()[:30]:
    print(" -", career)

# ============================================================
# 7. HANDLE MULTIPLE EXPECTED CAREERS
# ============================================================

def parse_expected(value):

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Python-list representation
    if text.startswith("[") and text.endswith("]"):

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]

        except Exception:
            pass

    # Common separators
    for separator in [
        "|",
        ";",
        ","
    ]:

        if separator in text:

            parts = [
                x.strip()
                for x in text.split(separator)
                if x.strip()
            ]

            if len(parts) > 1:
                return parts

    return [text]


# ============================================================
# 8. BUILD GROUND TRUTH
# ============================================================

ground_truth = []

for value in df[career_column]:

    careers = parse_expected(value)

    ground_truth.append(careers)

# ============================================================
# 9. CHECK V4 / DATASET COMPATIBILITY
# ============================================================

print("\n" + "=" * 70)
print("6. CAREER LABEL COMPATIBILITY")
print("=" * 70)

v4_careers = set(
    v4["career"]
    .astype(str)
    .str.strip()
)

ground_truth_careers = set()

for careers in ground_truth:

    for career in careers:
        ground_truth_careers.add(career)

intersection = (
    v4_careers
    &
    ground_truth_careers
)

print(
    "V4 career labels:",
    len(v4_careers)
)

print(
    "Ground-truth career labels:",
    len(ground_truth_careers)
)

print(
    "Common labels:",
    len(intersection)
)

if not intersection:

    print(
        "\n[WARNING]"
    )

    print(
        "The validation dataset and V4 output "
        "do not share career labels."
    )

    print(
        "\nTherefore Top-K accuracy cannot "
        "be calculated honestly."
    )

    print(
        "\nNo fake PASS will be generated."
    )

    raise SystemExit(0)

# ============================================================
# 10. TOP-K METRICS
# ============================================================

print("\n" + "=" * 70)
print("7. TOP-K VALIDATION")
print("=" * 70)

# The current V4 file is a single candidate's
# recommendation list. Therefore it cannot be
# directly evaluated against multiple candidates
# unless candidate IDs are present.

if "candidate_id" not in v4.columns:

    print(
        "\n[INFO]"
    )

    print(
        "V4 output does not contain candidate_id."
    )

    print(
        "Therefore this file represents one "
        "recommendation run, not a multi-candidate "
        "benchmark evaluation."
    )

    print(
        "\nWe will perform a label-level sanity "
        "check instead of inventing candidate mappings."
    )

    top5 = (
        v4
        .sort_values("rank")
        .head(5)["career"]
        .astype(str)
        .tolist()
    )

    print("\nCurrent V4 Top-5:")

    for i, career in enumerate(
        top5,
        start=1
    ):
        print(f"{i}. {career}")

    print("\nExpected presentation Top-5:")

    for i, career in enumerate(
        DEMO_CAREERS,
        start=1
    ):
        print(f"{i}. {career}")

    if top5 == DEMO_CAREERS:

        print(
            "\nPresentation Top-5 consistency: PASS"
        )

    else:

        print(
            "\nPresentation Top-5 consistency: FAIL"
        )

    print(
        "\nActual benchmark accuracy was NOT "
        "calculated because candidate-level "
        "ground truth is unavailable."
    )

    raise SystemExit(0)

# ============================================================
# 11. MULTI-CANDIDATE TOP-K
# ============================================================

print(
    "\nCandidate-level V4 validation detected."
)

results = []

for candidate_id, group in v4.groupby(
    "candidate_id"
):

    group = group.sort_values("rank")

    predictions = (
        group["career"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    matching_rows = df[
        df["candidate_id"].astype(str)
        == str(candidate_id)
    ]

    if matching_rows.empty:
        continue

    expected = parse_expected(
        matching_rows.iloc[0][career_column]
    )

    if not expected:
        continue

    results.append(
        {
            "candidate_id":
                candidate_id,
            "predictions":
                predictions,
            "expected":
                expected
        }
    )

if not results:

    print(
        "\nNo candidate-level matches found."
    )

    print(
        "No benchmark accuracy will be generated."
    )

    raise SystemExit(0)

# ============================================================
# 12. CALCULATE TOP-K
# ============================================================

def top_k_accuracy(
    records,
    k
):

    correct = 0

    for record in records:

        predictions = record[
            "predictions"
        ][:k]

        expected = set(
            record["expected"]
        )

        if any(
            prediction in expected
            for prediction in predictions
        ):
            correct += 1

    return (
        correct /
        len(records)
    )


top1 = top_k_accuracy(
    results,
    1
)

top3 = top_k_accuracy(
    results,
    3
)

top5 = top_k_accuracy(
    results,
    5
)

print("\n" + "=" * 70)
print("8. BENCHMARK RESULTS")
print("=" * 70)

print(
    f"Candidates evaluated: {len(results)}"
)

print(
    f"Top-1 Accuracy: {top1:.4f}"
)

print(
    f"Top-3 Accuracy: {top3:.4f}"
)

print(
    f"Top-5 Accuracy: {top5:.4f}"
)

# ============================================================
# 13. MRR
# ============================================================

reciprocal_ranks = []

for record in results:

    expected = set(
        record["expected"]
    )

    predictions = record[
        "predictions"
    ]

    rank_found = None

    for rank, prediction in enumerate(
        predictions,
        start=1
    ):

        if prediction in expected:
            rank_found = rank
            break

    if rank_found is None:
        reciprocal_ranks.append(0.0)

    else:
        reciprocal_ranks.append(
            1.0 / rank_found
        )

mrr = np.mean(
    reciprocal_ranks
)

print(
    f"MRR: {mrr:.4f}"
)

# ============================================================
# 14. SAVE RESULTS
# ============================================================

output_dir = os.path.join(
    PROJECT_ROOT,
    "results",
    "benchmark_validation"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

summary = {
    "validation_dataset":
        validation_path,

    "v4_output":
        v4_path,

    "candidates_evaluated":
        len(results),

    "top_1_accuracy":
        round(float(top1), 6),

    "top_3_accuracy":
        round(float(top3), 6),

    "top_5_accuracy":
        round(float(top5), 6),

    "mrr":
        round(float(mrr), 6),

    "presentation_demo": DEMO_CAREERS,

    "v4_modified": False,

    "models_modified": False,

    "dataset_modified": False
}

summary_path = os.path.join(
    output_dir,
    "benchmark_validation_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2
    )

print(
    "\nSummary saved:"
)

print(
    summary_path
)

# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("FINAL BENCHMARK STATUS")
print("=" * 70)

print(
    "\nImportant:"
)

print(
    "This validation uses the existing "
    "CareerCast curated validation dataset."
)

print(
    "It is NOT being falsely labeled as "
    "SemEval or LinkedIn."
)

print(
    "\nPresentation demo remains:"
)

for i, career in enumerate(
    DEMO_CAREERS,
    start=1
):
    print(
        f"{i}. {career}"
    )

print(
    "\nV4 recommender: NOT MODIFIED"
)

print(
    "Models: NOT MODIFIED"
)

print(
    "Dataset: NOT MODIFIED"
)

print(
    "\nBenchmark validation completed "
    "only where ground truth supports it."
)

print(
    "=" * 70
)
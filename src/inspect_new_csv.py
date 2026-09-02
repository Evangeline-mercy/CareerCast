import os
import pandas as pd

CSV_PATH = "datasets/global_ai_jobs_dataset.csv"

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("NEW CSV INSPECTION")
print("=" * 80)

print("\nFile:")
print(CSV_PATH)

if not os.path.exists(CSV_PATH):
    print("\nERROR: CSV file not found.")
    print("Check the filename and datasets folder.")
    raise SystemExit

print("\nLoading CSV...")

df = pd.read_csv(
    CSV_PATH,
    low_memory=False
)

print("\n" + "=" * 80)
print("1. BASIC INFORMATION")
print("=" * 80)

print("Rows       :", len(df))
print("Columns    :", len(df.columns))

print("\n" + "=" * 80)
print("2. COLUMN NAMES")
print("=" * 80)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n" + "=" * 80)
print("3. DATA TYPES")
print("=" * 80)

print(df.dtypes)

print("\n" + "=" * 80)
print("4. MISSING VALUES")
print("=" * 80)

missing = df.isnull().sum()

for column, count in missing.items():
    print(f"{column}: {count}")

print("\n" + "=" * 80)
print("5. SAMPLE ROWS")
print("=" * 80)

print(df.head(5).to_string())

print("\n" + "=" * 80)
print("6. UNIQUE VALUES")
print("=" * 80)

for column in df.columns:

    if df[column].dtype == "object":

        unique_count = df[column].nunique()

        print(
            f"{column}: {unique_count} unique values"
        )

print("\n" + "=" * 80)
print("7. POSSIBLE CAREER / JOB COLUMNS")
print("=" * 80)

keywords = [
    "career",
    "job",
    "title",
    "occupation",
    "role",
    "position"
]

for column in df.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in keywords
    ):

        print(
            f"\nCOLUMN: {column}"
        )

        print(
            df[column]
            .dropna()
            .astype(str)
            .head(10)
            .tolist()
        )

print("\n" + "=" * 80)
print("8. POSSIBLE SKILL COLUMNS")
print("=" * 80)

skill_keywords = [
    "skill",
    "skills",
    "technology",
    "technologies",
    "tools",
    "software"
]

for column in df.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in skill_keywords
    ):

        print(
            f"\nCOLUMN: {column}"
        )

        print(
            df[column]
            .dropna()
            .astype(str)
            .head(10)
            .tolist()
        )

print("\n" + "=" * 80)
print("9. POSSIBLE JOB DESCRIPTION COLUMNS")
print("=" * 80)

description_keywords = [
    "description",
    "summary",
    "responsibilities",
    "requirements"
]

for column in df.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in description_keywords
    ):

        print(
            f"\nCOLUMN: {column}"
        )

        print(
            df[column]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
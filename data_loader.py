import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "datasets"

files = [
    "career_interest_types.csv",
    "education.csv",
    "essential_skills.csv",
    "job_titles.csv",
    "occupation_data.csv",
    "software_skills.csv",
    "training_and_experience.csv"
]

for file in files:
    path = DATA_DIR / file

    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
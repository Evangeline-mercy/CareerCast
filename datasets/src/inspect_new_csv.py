import pandas as pd
import os

CSV_PATH = "datasets/job_skill_set.csv"

print("=" * 70)
print("NEW CSV INSPECTION")
print("=" * 70)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("\n1. Shape")
print("Rows    :", len(df))
print("Columns :", len(df.columns))

print("\n2. Columns")
for column in df.columns:
    print("-", column)

print("\n3. Data types")
print(df.dtypes)

print("\n4. Missing values")
print(df.isnull().sum())

print("\n5. Duplicate rows")
print(df.duplicated().sum())

print("\n6. First 5 rows")
print(df.head().to_string())

print("\n7. Unique values")
for column in df.columns:
    if df[column].dtype == "object":
        print(
            f"{column}: "
            f"{df[column].nunique()} unique values"
        )

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
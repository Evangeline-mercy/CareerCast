import pandas as pd
from pathlib import Path

INPUT = Path(
    "results/milestone2_unified/milestone2_unified_career_dataset.csv"
)

OUTPUT = Path(
    "results/milestone2_unified/milestone2_balanced_dataset.csv"
)

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("BALANCED DATASET BUILDER")
print("=" * 90)

print("\n[1] Loading unified dataset...")

df = pd.read_csv(INPUT)

print("Rows loaded :", len(df))
print("Columns     :", list(df.columns))
print("Careers     :", df["career"].nunique())

print("\n" + "=" * 90)
print("ORIGINAL CAREER DISTRIBUTION")
print("=" * 90)

counts = df["career"].value_counts()
print(counts.to_string())

target_count = counts.min()

print("\nMinimum career count :", target_count)
print("Target samples/class :", target_count)

balanced_parts = []

for career in sorted(df["career"].unique()):

    career_df = df[df["career"] == career]

    sampled = career_df.sample(
        n=target_count,
        random_state=42
    )

    balanced_parts.append(sampled)

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

balanced_df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 90)
print("BALANCED CAREER DISTRIBUTION")
print("=" * 90)

print(
    balanced_df["career"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "=" * 90)
print("FINAL DATASET")
print("=" * 90)

print("Rows    :", len(balanced_df))
print("Columns :", len(balanced_df.columns))
print("Careers :", balanced_df["career"].nunique())

print("\nSaved:")
print(OUTPUT.resolve())

print("\n" + "=" * 90)
print("BALANCING COMPLETE")
print("=" * 90)

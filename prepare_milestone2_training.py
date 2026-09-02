import pandas as pd
from pathlib import Path

INPUT = Path(
    "results/milestone2_unified/milestone2_unified_career_dataset.csv"
)

OUTPUT = Path(
    "results/milestone2_unified/milestone2_training_dataset.csv"
)

RANDOM_STATE = 42

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("CONTROLLED TRAINING DATASET BUILDER")
print("=" * 90)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading unified dataset...")

df = pd.read_csv(INPUT)

df["career"] = df["career"].astype(str).str.strip()
df["skills"] = df["skills"].fillna("").astype(str)

print("Input rows    :", len(df))
print("Input careers :", df["career"].nunique())

# ============================================================
# 2. ORIGINAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("ORIGINAL DISTRIBUTION")
print("=" * 90)

counts = df["career"].value_counts()

print(counts.to_string())

# ============================================================
# 3. DEFINE CONTROLLED TARGET
# ============================================================

# Large AI classes:
# Keep up to 2000 real samples.
#
# Small classes:
# Oversample with replacement up to 500 samples.
#
# This avoids:
# - throwing away too much AI data
# - allowing AI classes to dominate completely
#
LARGE_CLASS_LIMIT = 2000
SMALL_CLASS_TARGET = 500

parts = []

print("\n" + "=" * 90)
print("BUILDING CONTROLLED TRAINING DATASET")
print("=" * 90)

for career in sorted(counts.index):

    career_df = df[df["career"] == career].copy()
    n = len(career_df)

    if n >= LARGE_CLASS_LIMIT:

        # Downsample large classes to 2000
        sampled = career_df.sample(
            n=LARGE_CLASS_LIMIT,
            random_state=RANDOM_STATE
        )

        method = f"downsampled {n} -> {LARGE_CLASS_LIMIT}"

    else:

        # Oversample small classes.
        # replace=True means rows can be repeated.
        sampled = career_df.sample(
            n=SMALL_CLASS_TARGET,
            replace=True,
            random_state=RANDOM_STATE
        )

        method = f"oversampled {n} -> {SMALL_CLASS_TARGET}"

    parts.append(sampled)

    print(
        f"{career:<32} "
        f"{n:>6} original -> "
        f"{len(sampled):>6} training | {method}"
    )

# ============================================================
# 4. COMBINE
# ============================================================

training_df = pd.concat(
    parts,
    ignore_index=True
)

# ============================================================
# 5. SHUFFLE
# ============================================================

training_df = training_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ============================================================
# 6. SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

training_df.to_csv(
    OUTPUT,
    index=False
)

# ============================================================
# 7. FINAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("FINAL TRAINING DISTRIBUTION")
print("=" * 90)

final_counts = (
    training_df["career"]
    .value_counts()
    .sort_index()
)

print(final_counts.to_string())

# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FINAL DATASET SUMMARY")
print("=" * 90)

print("Rows       :", len(training_df))
print("Columns    :", len(training_df.columns))
print("Careers    :", training_df["career"].nunique())
print("Min/class  :", final_counts.min())
print("Max/class  :", final_counts.max())

print("\nSaved:")
print(OUTPUT.resolve())

print("\n" + "=" * 90)
print("TRAINING DATASET PREPARATION COMPLETE")
print("=" * 90)
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

SOURCE = Path(
    "results/milestone2_training/"
    "career_profile_training_dataset.csv"
)

OUTPUT_DIR = Path("results/milestone2_validation")

RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("CAREERCAST - STRATIFIED 80/10/10 DATA SPLIT")
print("=" * 90)

df = pd.read_csv(SOURCE)

print("\nOriginal dataset:")
print("Rows:", len(df))
print("Careers:", df["career"].nunique())

# ============================================================
# FIRST SPLIT
# 80% TRAIN
# 20% TEMP
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["career"],
    random_state=RANDOM_STATE,
)

# ============================================================
# SECOND SPLIT
# TEMP -> 10% VALIDATION + 10% TEST
# ============================================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["career"],
    random_state=RANDOM_STATE,
)

# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train_path = OUTPUT_DIR / "train.csv"
validation_path = OUTPUT_DIR / "validation.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print("\nSPLIT RESULTS")
print("-" * 90)

print(
    f"Training   : {len(train_df):5d} rows "
    f"({len(train_df) / len(df) * 100:.1f}%)"
)

print(
    f"Validation : {len(validation_df):5d} rows "
    f"({len(validation_df) / len(df) * 100:.1f}%)"
)

print(
    f"Test       : {len(test_df):5d} rows "
    f"({len(test_df) / len(df) * 100:.1f}%)"
)

print(
    f"Total      : {len(train_df) + len(validation_df) + len(test_df):5d}"
)

print("\nSamples per career")
print("-" * 90)

print(
    "TRAIN:"
)
print(
    train_df["career"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nVALIDATION:")
print(
    validation_df["career"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTEST:")
print(
    test_df["career"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nSaved files:")
print(train_path)
print(validation_path)
print(test_path)

print("\nSPLIT COMPLETED")
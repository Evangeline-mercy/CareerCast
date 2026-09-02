import pandas as pd
import os
import random

INPUT_FILE = "results/milestone2_unified/career_profiles_96.csv"

OUTPUT_DIR = "results/milestone2_training"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "career_profile_training_dataset.csv"
)

SAMPLES_PER_CAREER = 500
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("PROFILE-BASED TRAINING DATASET BUILDER")
print("=" * 90)

print("\n[1] Loading clean career profiles...")

df = pd.read_csv(INPUT_FILE)

print("Careers loaded :", len(df))

print("\n[2] Generating controlled skill variations...")

records = []

for _, row in df.iterrows():

    career = row["job_title"]

    base_skills = [
        s.strip().lower()
        for s in str(row["required_skills"]).split("|")
        if s.strip()
    ]

    base_skills = list(dict.fromkeys(base_skills))

    for _ in range(SAMPLES_PER_CAREER):

        remove_count = random.randint(2, 5)

        if len(base_skills) > remove_count:
            selected = random.sample(
                base_skills,
                len(base_skills) - remove_count
            )
        else:
            selected = base_skills.copy()

        random.shuffle(selected)

        records.append({
            "career": career,
            "skills": " | ".join(selected)
        })

training_df = pd.DataFrame(records)

training_df = training_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

print("\n" + "=" * 90)
print("TRAINING DATASET QUALITY CHECK")
print("=" * 90)

print("Rows       :", len(training_df))
print("Careers    :", training_df["career"].nunique())

counts = training_df["career"].value_counts()

print("\nCAREER DISTRIBUTION")
print(counts.sort_index().to_string())

print("\nMinimum samples/class :", counts.min())
print("Maximum samples/class :", counts.max())

empty_skills = (
    training_df["skills"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

print("\nEmpty skill profiles :", empty_skills)

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 90)
print("FINAL DATASET")
print("=" * 90)

print("Rows       :", len(training_df))
print("Columns    :", len(training_df.columns))
print("Careers    :", training_df["career"].nunique())

print("\nSaved:")
print(os.path.abspath(OUTPUT_FILE))

print("\n" + "=" * 90)
print("PROFILE-BASED TRAINING DATASET COMPLETE")
print("=" * 90)
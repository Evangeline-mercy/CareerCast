import pandas as pd
import os

INPUT = "it_jobs_required_skills.csv"

OUTPUT_DIR = "results/milestone2_clean"
OUTPUT = os.path.join(
    OUTPUT_DIR,
    "clean_87_career_profiles.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("CLEAN CAREER PROFILE BUILDER")
print("=" * 90)

df = pd.read_csv(INPUT)

# Basic cleaning
df["job_title"] = (
    df["job_title"]
    .astype(str)
    .str.strip()
)

df["required_skills"] = (
    df["required_skills"]
    .fillna("")
    .astype(str)
)

# Normalize skill formatting
def normalize_skills(text):
    skills = []

    for skill in text.split("|"):
        skill = skill.strip().lower()

        if skill and skill not in skills:
            skills.append(skill)

    return " | ".join(skills)

df["skills"] = df["required_skills"].apply(normalize_skills)

# Keep only required columns
clean = df[["job_title", "skills"]].copy()

# Rename career column
clean = clean.rename(
    columns={"job_title": "career"}
)

# Remove duplicate career labels
clean = clean.drop_duplicates(
    subset=["career"]
)

print()
print("Total careers :", len(clean))
print("Unique careers:", clean["career"].nunique())

skill_counts = clean["skills"].str.split("|").str.len()

print("Minimum skills :", skill_counts.min())
print("Maximum skills :", skill_counts.max())
print("Average skills :", round(skill_counts.mean(), 2))

print()
print("CAREERS")
print("-" * 90)

for career in clean["career"]:
    print(career)

clean.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 90)
print("CLEAN DATASET SAVED")
print("=" * 90)
print(OUTPUT)
print()
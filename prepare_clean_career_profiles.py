import pandas as pd
import os

INPUT_FILE = "it_jobs_required_skills.csv"
OUTPUT_DIR = "results/milestone2_unified"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clean_career_profiles.csv")

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("CLEAN CAREER PROFILE PREPARATION")
print("=" * 90)

# ------------------------------------------------------------------
# 1. Load source dataset
# ------------------------------------------------------------------

print("\n[1] Loading career profile dataset...")

df = pd.read_csv(INPUT_FILE)

print("Rows loaded :", len(df))
print("Columns     :", list(df.columns))

# ------------------------------------------------------------------
# 2. Basic cleaning
# ------------------------------------------------------------------

print("\n[2] Cleaning career profiles...")

df = df[["job_title", "required_skills"]].copy()

df["job_title"] = (
    df["job_title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["required_skills"] = (
    df["required_skills"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

# Remove empty rows
df = df[
    (df["job_title"] != "") &
    (df["required_skills"] != "")
].copy()

# Remove duplicate career names
df = df.drop_duplicates(
    subset=["job_title"],
    keep="first"
).reset_index(drop=True)

# ------------------------------------------------------------------
# 3. Clean skill lists
# ------------------------------------------------------------------

def clean_skills(skill_string):

    skills = skill_string.split("|")

    cleaned = []

    for skill in skills:

        skill = skill.strip().lower()

        if skill and skill not in cleaned:
            cleaned.append(skill)

    return " | ".join(cleaned)


df["required_skills"] = df["required_skills"].apply(clean_skills)

# ------------------------------------------------------------------
# 4. Skill count
# ------------------------------------------------------------------

df["skill_count"] = (
    df["required_skills"]
    .str.split("|")
    .str.len()
)

# ------------------------------------------------------------------
# 5. Quality check
# ------------------------------------------------------------------

print("\n" + "=" * 90)
print("CAREER PROFILE QUALITY CHECK")
print("=" * 90)

print("Total careers :", len(df))
print("Unique careers:", df["job_title"].nunique())
print("Minimum skills:", df["skill_count"].min())
print("Maximum skills:", df["skill_count"].max())
print("Average skills:", round(df["skill_count"].mean(), 2))

# ------------------------------------------------------------------
# 6. Validate 20 skills per career
# ------------------------------------------------------------------

invalid = df[df["skill_count"] != 20]

print("\nProfiles with skill count != 20 :", len(invalid))

if len(invalid) > 0:

    print("\nWARNING: These careers do not contain exactly 20 skills:")

    print(
        invalid[
            ["job_title", "skill_count"]
        ].to_string(index=False)
    )

else:

    print("All career profiles contain exactly 20 skills.")

# ------------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_df = df[
    ["job_title", "required_skills", "skill_count"]
].copy()

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------------
# 8. Final summary
# ------------------------------------------------------------------

print("\n" + "=" * 90)
print("FINAL DATASET SUMMARY")
print("=" * 90)

print("Total careers :", len(output_df))
print("Unique careers:", output_df["job_title"].nunique())
print("Minimum skills:", output_df["skill_count"].min())
print("Maximum skills:", output_df["skill_count"].max())
print("Average skills:", round(output_df["skill_count"].mean(), 2))

print("\nSaved:")
print(os.path.abspath(OUTPUT_FILE))

print("\n" + "=" * 90)
print("CLEAN CAREER PROFILE PREPARATION COMPLETE")
print("=" * 90)

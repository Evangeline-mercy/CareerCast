import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CareerCast Milestone 2 - Fresh Dataset Builder
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)

print("=" * 75)
print("CAREERCAST MILESTONE 2 - DATASET BUILD")
print("=" * 75)

# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

files = {
    "occupation": "occupation_data.csv",
    "skills": "essential_skills.csv",
    "software": "software_skills.csv",
    "education": "education.csv",
    "experience": "training_and_experience.csv",
    "interest": "career_interest_types.csv",
}

dfs = {}

for key, filename in files.items():
    path = DATA_DIR / filename

    print(f"\nLoading: {filename}")

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    dfs[key] = df

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# Find common occupation codes
# ------------------------------------------------------------

code_sets = [
    set(df["O*NET-SOC Code"].dropna().astype(str).str.strip())
    for df in dfs.values()
]

common_codes = set.intersection(*code_sets)

print("\n" + "=" * 75)
print("COMMON OCCUPATION ANALYSIS")
print("=" * 75)

print(f"Common O*NET-SOC codes: {len(common_codes)}")


# ------------------------------------------------------------
# Base occupation table
# ------------------------------------------------------------

occupation = dfs["occupation"].copy()

occupation["O*NET-SOC Code"] = (
    occupation["O*NET-SOC Code"]
    .astype(str)
    .str.strip()
)

occupation = occupation[
    occupation["O*NET-SOC Code"].isin(common_codes)
].copy()

occupation = occupation[
    ["O*NET-SOC Code", "Title", "Description"]
].drop_duplicates(
    subset=["O*NET-SOC Code"]
)

occupation = occupation.rename(
    columns={
        "Title": "Career",
        "Description": "Job_Description"
    }
)

print(f"Base occupations: {len(occupation):,}")


# ============================================================
# ESSENTIAL SKILLS
# ============================================================

skills = dfs["skills"].copy()

skills["O*NET-SOC Code"] = (
    skills["O*NET-SOC Code"]
    .astype(str)
    .str.strip()
)

skills = skills[
    skills["O*NET-SOC Code"].isin(common_codes)
].copy()

# Use Importance records when available.
# Keep the skill name and importance value.

skills["Element Name"] = (
    skills["Element Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

if "Scale ID" in skills.columns:
    importance = skills[
        skills["Scale ID"].astype(str).str.upper().eq("IM")
    ].copy()
else:
    importance = skills.copy()

if len(importance) == 0:
    importance = skills.copy()

importance["Data Value"] = pd.to_numeric(
    importance["Data Value"],
    errors="coerce"
)

# Remove invalid rows
importance = importance[
    importance["Element Name"].ne("")
].copy()

# Sort important skills first
importance = importance.sort_values(
    ["O*NET-SOC Code", "Data Value"],
    ascending=[True, False]
)

# Keep top 15 essential skills per occupation
top_skills = (
    importance
    .groupby("O*NET-SOC Code", group_keys=False)
    .head(15)
)

skill_text = (
    top_skills
    .groupby("O*NET-SOC Code")["Element Name"]
    .apply(lambda x: ", ".join(dict.fromkeys(x)))
    .rename("Essential_Skills")
    .reset_index()
)

print(f"Occupations with essential skills: {len(skill_text):,}")


# ============================================================
# SOFTWARE SKILLS
# ============================================================

software = dfs["software"].copy()

software["O*NET-SOC Code"] = (
    software["O*NET-SOC Code"]
    .astype(str)
    .str.strip()
)

software = software[
    software["O*NET-SOC Code"].isin(common_codes)
].copy()

software["Workplace Example"] = (
    software["Workplace Example"]
    .fillna("")
    .astype(str)
    .str.strip()
)

software["Element Name"] = (
    software["Element Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Prefer workplace examples because they provide
# concrete software/tool names.

software["Software"] = np.where(
    software["Workplace Example"].ne(""),
    software["Workplace Example"],
    software["Element Name"]
)

software = software[software["Software"].ne("")].copy()

# Remove duplicate software names per occupation
software = software.drop_duplicates(
    subset=["O*NET-SOC Code", "Software"]
)

software_text = (
    software
    .groupby("O*NET-SOC Code")["Software"]
    .apply(lambda x: ", ".join(x.head(20)))
    .rename("Software_Skills")
    .reset_index()
)

print(f"Occupations with software skills: {len(software_text):,}")


# ============================================================
# EDUCATION
# ============================================================

education = dfs["education"].copy()

education["O*NET-SOC Code"] = (
    education["O*NET-SOC Code"]
    .astype(str)
    .str.strip()
)

education = education[
    education["O*NET-SOC Code"].isin(common_codes)
].copy()

education["Element Name"] = (
    education["Element Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

education["Data Value"] = pd.to_numeric(
    education["Data Value"],
    errors="coerce"
)

# Keep education-related records
education = education[
    education["Element Name"].str.contains(
        "education",
        case=False,
        na=False
    )
].copy()

# Calculate weighted/average education requirement
education_level = (
    education
    .groupby("O*NET-SOC Code")["Data Value"]
    .mean()
    .rename("Education_Level")
    .reset_index()
)

print(f"Occupations with education data: {len(education_level):,}")


# ============================================================
# EXPERIENCE
# ============================================================

experience = dfs["experience"].copy()

experience["O*NET-SOC Code"] = (
    experience["O*NET-SOC Code"]
    .astype(str)
    .str.strip()
)

experience = experience[
    experience["O*NET-SOC Code"].isin(common_codes)
].copy()

experience["Element Name"] = (
    experience["Element Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

experience["Data Value"] = pd.to_numeric(
    experience["Data Value"],
    errors="coerce"
)

experience = experience[
    experience["Element Name"].str.contains(
        "experience",
        case=False,
        na=False
    )
].copy()

experience_value = (
    experience
    .groupby("O*NET-SOC Code")["Data Value"]
    .mean()
    .rename("Experience_Level")
    .reset_index()
)

print(f"Occupations with experience data: {len(experience_value):,}")


# ============================================================
# CAREER INTEREST
# ============================================================

interest = dfs["interest"].copy()

interest["O*NET-SOC Code"] = (
    interest["O*NET-SOC Code"]
    .astype(str)
    .str.strip()
)

interest = interest[
    interest["O*NET-SOC Code"].isin(common_codes)
].copy()

interest["Element Name"] = (
    interest["Element Name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

interest["Data Value"] = pd.to_numeric(
    interest["Data Value"],
    errors="coerce"
)

# Convert interest types into:
# Realistic / Investigative / Artistic / Social /
# Enterprising / Conventional

interest_pivot = (
    interest
    .pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean"
    )
    .reset_index()
)

interest_columns = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional"
]

for col in interest_columns:
    if col not in interest_pivot.columns:
        interest_pivot[col] = 0.0

interest_pivot = interest_pivot[
    ["O*NET-SOC Code"] + interest_columns
]

print(
    f"Occupations with interest data: "
    f"{len(interest_pivot):,}"
)


# ============================================================
# MERGE EVERYTHING
# ============================================================

dataset = occupation.copy()

dataset = dataset.merge(
    skill_text,
    on="O*NET-SOC Code",
    how="left"
)

dataset = dataset.merge(
    software_text,
    on="O*NET-SOC Code",
    how="left"
)

dataset = dataset.merge(
    education_level,
    on="O*NET-SOC Code",
    how="left"
)

dataset = dataset.merge(
    experience_value,
    on="O*NET-SOC Code",
    how="left"
)

dataset = dataset.merge(
    interest_pivot,
    on="O*NET-SOC Code",
    how="left"
)


# ============================================================
# CLEAN
# ============================================================

text_columns = [
    "Career",
    "Job_Description",
    "Essential_Skills",
    "Software_Skills"
]

for col in text_columns:
    dataset[col] = (
        dataset[col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

numeric_columns = [
    "Education_Level",
    "Experience_Level"
] + interest_columns

for col in numeric_columns:
    dataset[col] = pd.to_numeric(
        dataset[col],
        errors="coerce"
    )

    dataset[col] = dataset[col].fillna(
        dataset[col].median()
    )


# ------------------------------------------------------------
# Combined text feature
# ------------------------------------------------------------

dataset["Combined_Text"] = (
    "Career: "
    + dataset["Career"]
    + " Skills: "
    + dataset["Essential_Skills"]
    + " Software: "
    + dataset["Software_Skills"]
    + " Description: "
    + dataset["Job_Description"]
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

columns = [
    "O*NET-SOC Code",
    "Career",
    "Essential_Skills",
    "Software_Skills",
    "Job_Description",
    "Education_Level",
    "Experience_Level",
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
    "Combined_Text"
]

dataset = dataset[columns]


# ============================================================
# SAVE
# ============================================================

output_path = (
    RESULT_DIR / "careercast_milestone2_dataset.csv"
)

dataset.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 75)
print("FINAL DATASET")
print("=" * 75)

print("Rows:", len(dataset))
print("Columns:", len(dataset.columns))

print("\nColumns:")
for col in dataset.columns:
    print(" -", col)

print("\nMissing values:")
print(
    dataset.isnull().sum().to_string()
)

print("\nFirst 5 records:")
print(
    dataset.head(5).to_string(index=False)
)

print("\nCareer count:")
print(
    dataset["Career"].nunique()
)

print("\nSaved to:")
print(output_path)

print("=" * 75)
print("DATASET BUILD COMPLETE")
print("=" * 75)
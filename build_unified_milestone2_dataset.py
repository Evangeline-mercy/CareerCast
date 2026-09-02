import pandas as pd
import os
import re

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("UNIFIED CAREER DATASET BUILDER")
print("=" * 90)

BASE = os.getcwd()
OUTPUT = os.path.join(BASE, "results", "milestone2_unified")

os.makedirs(OUTPUT, exist_ok=True)


# ============================================================
# 1. CAREER ALIAS NORMALIZATION
# ============================================================

ALIASES = {
    # AI / ML
    "data scientist": "Data Scientist",
    "data scientists": "Data Scientist",

    "machine learning engineer": "Machine Learning Engineer",
    "machine learning engineers": "Machine Learning Engineer",

    "ai engineer": "AI Engineer",
    "artificial intelligence engineer": "AI Engineer",

    "mlops engineer": "MLOps Engineer",
    "ml ops engineer": "MLOps Engineer",

    "nlp engineer": "NLP Engineer",

    "computer vision engineer": "Computer Vision Engineer",

    "ai researcher": "AI Researcher",
    "computer and information research scientist":
        "AI Researcher",
    "computer and information research scientists":
        "AI Researcher",

    "prompt engineer": "Prompt Engineer",
    "ai prompt engineer": "Prompt Engineer",

    "ai product manager": "AI Product Manager",

    # DATA
    "data analyst": "Data Analyst",
    "data analysts": "Data Analyst",

    "data engineer": "Data Engineer",

    "business intelligence developer":
        "Business Intelligence Developer",

    "database administrator":
        "Database Administrator",

    "quantitative analyst":
        "Quantitative Analyst",

    # SOFTWARE
    "software developer": "Software Developer",
    "software developers": "Software Developer",

    "software engineer": "Software Developer",
    "senior software engineer": "Software Developer",

    "backend developer": "Backend Developer",
    "backend developers": "Backend Developer",

    "frontend developer": "Frontend Developer",
    "frontend developers": "Frontend Developer",

    "fullstack developer": "Full Stack Developer",
    "full stack developer": "Full Stack Developer",

    "python developer": "Python Developer",
    "java developer": "Java Developer",

    ".net developer": ".NET Developer",

    "react developer": "React Developer",

    # CLOUD / DEVOPS
    "cloud engineer": "Cloud Engineer",
    "cloud engineers": "Cloud Engineer",

    "devops engineer": "DevOps Engineer",

    "site reliability engineer":
        "Site Reliability Engineer",

    # EMBEDDED / ECE
    "embedded systems engineer":
        "Embedded Systems Engineer",

    "embedded systems engineers":
        "Embedded Systems Engineer",

    "firmware engineer": "Firmware Engineer",
    "firmware engineers": "Firmware Engineer",

    "iot engineer": "IoT Engineer",
    "iot engineers": "IoT Engineer",

    "vlsi engineer": "VLSI Engineer",
    "vlsi engineers": "VLSI Engineer",

    "electronics engineer": "Electronics Engineer",
    "electronics engineers": "Electronics Engineer",

    "signal processing engineer":
        "Signal Processing Engineer",

    # WEB
    "web developer": "Web Developer",
    "web developers": "Web Developer",

    # OTHER
    "operations research analyst":
        "Operations Research Analyst",

    "statistician": "Statistician",

    "electrical engineer": "Electrical Engineer",

    "computer hardware engineer":
        "Computer Hardware Engineer",

    "blockchain developer":
        "Blockchain Developer",

    "game developer":
        "Game Developer",

    "ethical hacker":
        "Ethical Hacker",
}


def normalize_career(title):
    if pd.isna(title):
        return None

    x = str(title).strip().lower()

    # remove repeated whitespace
    x = re.sub(r"\s+", " ", x)

    return ALIASES.get(x, None)


def clean_skills(value):
    if pd.isna(value):
        return ""

    text = str(value)

    # Convert all major delimiters into |
    text = text.replace(",", "|")
    text = text.replace(";", "|")
    text = text.replace("/", "|")

    parts = []

    for item in text.split("|"):
        item = item.strip().lower()

        if item:
            parts.append(item)

    # remove duplicates while preserving order
    seen = set()
    result = []

    for item in parts:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return " | ".join(result)


records = []


# ============================================================
# 2. IT REQUIRED SKILLS DATASET
# ============================================================

print("\n[1/4] Loading it_jobs_required_skills.csv")

path = "it_jobs_required_skills.csv"

if os.path.exists(path):

    df = pd.read_csv(path)

    for _, row in df.iterrows():

        career = normalize_career(row["job_title"])

        if career is None:
            continue

        records.append({
            "career": career,
            "skills": clean_skills(row["required_skills"]),
            "source": "it_jobs_required_skills"
        })

    print("Rows loaded:", len(df))


# ============================================================
# 3. JOB DATASET
# ============================================================

print("\n[2/4] Loading job_dataset.csv")

path = "job_dataset.csv"

if os.path.exists(path):

    df = pd.read_csv(path)

    for _, row in df.iterrows():

        career = normalize_career(row["Title"])

        if career is None:
            continue

        # Combine Skills + Keywords
        skills = clean_skills(
            str(row.get("Skills", "")) +
            " | " +
            str(row.get("Keywords", ""))
        )

        records.append({
            "career": career,
            "skills": skills,
            "source": "job_dataset"
        })

    print("Rows loaded:", len(df))


# ============================================================
# 4. EXTENDED CLASSIFICATION DATASET
# ============================================================

print("\n[3/4] Loading job_titles_classification_extended.csv")

path = "job_titles_classification_extended.csv"

if os.path.exists(path):

    df = pd.read_csv(path)

    for _, row in df.iterrows():

        career = normalize_career(row["job_title"])

        if career is None:
            continue

        records.append({
            "career": career,
            "skills": clean_skills(row["skills_required"]),
            "source": "job_titles_classification_extended"
        })

    print("Rows loaded:", len(df))


# ============================================================
# 5. EXISTING GLOBAL AI DATASET
# ============================================================

print("\n[4/4] Loading global_ai_jobs_dataset.csv")

path = os.path.join(
    "datasets",
    "global_ai_jobs_dataset.csv"
)

if os.path.exists(path):

    df = pd.read_csv(path)

    for _, row in df.iterrows():

        career = normalize_career(row["job_title"])

        if career is None:
            continue

        records.append({
            "career": career,
            "skills": clean_skills(row["skills"]),
            "source": "global_ai_jobs_dataset"
        })

    print("Rows loaded:", len(df))


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

print("\n" + "=" * 90)
print("BUILDING UNIFIED DATASET")
print("=" * 90)

unified = pd.DataFrame(records)

print("\nRaw unified rows:", len(unified))

# Remove empty skills
unified = unified[
    unified["skills"].fillna("").str.strip() != ""
].copy()

# Remove duplicates
unified = unified.drop_duplicates(
    subset=["career", "skills"]
).reset_index(drop=True)

print("After duplicate removal:", len(unified))


# ============================================================
# 7. CAREER DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("CAREER DISTRIBUTION")
print("=" * 90)

counts = (
    unified["career"]
    .value_counts()
    .sort_values(ascending=False)
)

print(counts.to_string())


# ============================================================
# 8. KEEP TECHNICAL CAREERS WITH SUFFICIENT DATA
# ============================================================

MIN_ROWS = 20

valid_careers = counts[
    counts >= MIN_ROWS
].index.tolist()

final_df = unified[
    unified["career"].isin(valid_careers)
].copy()

print("\nMinimum rows per career:", MIN_ROWS)
print("Careers retained:", len(valid_careers))
print("Final rows:", len(final_df))


# ============================================================
# 9. SAVE
# ============================================================

output_csv = os.path.join(
    OUTPUT,
    "milestone2_unified_career_dataset.csv"
)

final_df.to_csv(
    output_csv,
    index=False
)

print("\nSaved:")
print(output_csv)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FINAL DATASET SUMMARY")
print("=" * 90)

print("Rows    :", len(final_df))
print("Columns :", len(final_df.columns))
print("Careers :", final_df["career"].nunique())

print("\nCareers:")

for career in sorted(final_df["career"].unique()):
    n = (final_df["career"] == career).sum()
    print(f" - {career}: {n}")

print("\n" + "=" * 90)
print("UNIFIED DATASET BUILD COMPLETE")
print("=" * 90)
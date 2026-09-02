import os
import json
import re
from datetime import datetime

import pandas as pd
import numpy as np


# ================================================================
# CAREERCAST MILESTONE 2
# FINAL CAREER RECOMMENDATION ENGINE
# NEW CSV + SKILL MATCHING + CAREER REPORT
# ================================================================

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("FINAL CAREER RECOMMENDATION ENGINE")
print("=" * 80)


# ================================================================
# 1. PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NEW_CSV = os.path.join(
    BASE_DIR,
    "datasets",
    "global_ai_jobs_dataset.csv"
)

OUTPUT_BASE = os.path.join(
    BASE_DIR,
    "v4_output"
)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = os.path.join(
    OUTPUT_BASE,
    "final_recommendation_" + RUN_ID
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "final_recommendations.csv"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "final_recommendations.json"
)

REPORT_TXT = os.path.join(
    OUTPUT_DIR,
    "career_report.txt"
)


# ================================================================
# 2. CANDIDATE PROFILE
# ================================================================

candidate_skills = [
    "python",
    "sql",
    "machine learning",
    "pandas",
    "numpy",
    "data analysis",
    "tensorflow"
]

candidate_skills = [s.lower().strip() for s in candidate_skills]

candidate_profile = {
    "education": "B.E. Electronics and Communication Engineering",
    "experience": "6 months internship experience",
    "skills": candidate_skills
}


print("\n" + "=" * 80)
print("1. CANDIDATE PROFILE")
print("=" * 80)

print("Education :", candidate_profile["education"])
print("Experience:", candidate_profile["experience"])

print("\nSkills:")
for skill in candidate_skills:
    print(" -", skill)


# ================================================================
# 3. LOAD NEW CSV
# ================================================================

print("\n" + "=" * 80)
print("2. LOADING NEW CSV DATASET")
print("=" * 80)

if not os.path.exists(NEW_CSV):
    print("\nERROR: Dataset not found:")
    print(NEW_CSV)
    raise SystemExit(1)

df = pd.read_csv(NEW_CSV)

print("Rows    :", len(df))
print("Columns :", len(df))

required_columns = [
    "job_title",
    "skills",
    "tools_used"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    print(missing_columns)
    raise SystemExit(1)

print("Required columns: PASS")


# ================================================================
# 4. NORMALIZATION FUNCTIONS
# ================================================================

def normalize_skill(skill):
    if pd.isna(skill):
        return ""

    skill = str(skill).lower().strip()

    skill = skill.replace("-", " ")
    skill = skill.replace("_", " ")

    skill = re.sub(r"\s+", " ", skill)

    return skill


def parse_skills(value):
    if pd.isna(value):
        return set()

    parts = str(value).split(";")

    return {
        normalize_skill(x)
        for x in parts
        if normalize_skill(x)
    }


# ================================================================
# 5. BUILD CAREER PROFILES
# ================================================================

print("\n" + "=" * 80)
print("3. BUILDING CAREER PROFILES")
print("=" * 80)

career_profiles = {}

for _, row in df.iterrows():

    career = str(row["job_title"]).strip()

    if not career:
        continue

    career_key = career.lower()

    skills = parse_skills(row["skills"])
    tools = parse_skills(row["tools_used"])

    combined_skills = skills.union(tools)

    if career_key not in career_profiles:
        career_profiles[career_key] = {
            "career": career,
            "skills": set(),
            "job_count": 0
        }

    career_profiles[career_key]["skills"].update(combined_skills)
    career_profiles[career_key]["job_count"] += 1


print("Unique careers:", len(career_profiles))


# ================================================================
# 6. CAREER SCORING
# ================================================================

print("\n" + "=" * 80)
print("4. CALCULATING CAREER MATCHING")
print("=" * 80)

results = []

for career_key, profile in career_profiles.items():

    career = profile["career"]
    career_skills = profile["skills"]

    matched_skills = sorted(
        set(candidate_skills).intersection(career_skills)
    )

    missing_skills = sorted(
        career_skills.difference(candidate_skills)
    )

    if len(candidate_skills) > 0:
        skill_alignment = (
            len(matched_skills) /
            len(candidate_skills)
        ) * 100
    else:
        skill_alignment = 0.0

    # Main candidate-to-career score
    match_score = skill_alignment / 100.0

    # Small bonus for number of available jobs
    # This prevents job count from dominating the score.
    job_count = profile["job_count"]

    job_bonus = min(
        np.log1p(job_count) / 20.0,
        0.05
    )

    final_score = min(
        match_score * 0.95 + job_bonus,
        1.0
    )

    results.append({
        "career": career,
        "final_score": round(final_score, 4),
        "skill_alignment": round(skill_alignment, 2),
        "matched_skills": ", ".join(matched_skills)
            if matched_skills else "None",
        "missing_skills": ", ".join(missing_skills[:15])
            if missing_skills else "None",
        "matched_skill_count": len(matched_skills),
        "candidate_skill_count": len(candidate_skills),
        "job_count": job_count
    })


# ================================================================
# 7. SORT RESULTS
# ================================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["final_score", "skill_alignment", "job_count"],
    ascending=False
).reset_index(drop=True)

results_df["rank"] = (
    results_df.index + 1
)

# Top 10 only
top10 = results_df.head(10).copy()


# ================================================================
# 8. DISPLAY TOP 10
# ================================================================

print("\n" + "=" * 80)
print("5. TOP 10 CAREER RECOMMENDATIONS")
print("=" * 80)

for _, row in top10.iterrows():

    print(
        f"\n{int(row['rank'])}. {row['career']}"
    )

    print(
        f"   Career Score      : "
        f"{row['final_score']:.4f}"
    )

    print(
        f"   Skill Alignment   : "
        f"{row['skill_alignment']:.2f}%"
    )

    print(
        f"   Matched Skills    : "
        f"{row['matched_skills']}"
    )

    print(
        f"   Skill Gaps        : "
        f"{row['missing_skills']}"
    )

    print(
        f"   Dataset Job Count : "
        f"{int(row['job_count'])}"
    )


# ================================================================
# 9. SKILL GAP SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("6. CANDIDATE SKILL GAP ANALYSIS")
print("=" * 80)

all_required_skills = set()

for profile in career_profiles.values():
    all_required_skills.update(profile["skills"])

candidate_skill_set = set(candidate_skills)

skill_gaps = sorted(
    all_required_skills.difference(candidate_skill_set)
)

print("\nCandidate skills:")
for skill in candidate_skills:
    print(" [FOUND]", skill)

print("\nPotential additional skills:")
for skill in skill_gaps[:20]:
    print(" [GAP]", skill)


# ================================================================
# 10. SAVE CSV
# ================================================================

top10.to_csv(
    OUTPUT_CSV,
    index=False
)


# ================================================================
# 11. SAVE JSON
# ================================================================

json_output = {
    "project": "CareerCast",
    "milestone": "Milestone 2",
    "generated_at": datetime.now().isoformat(),

    "candidate": {
        "education": candidate_profile["education"],
        "experience": candidate_profile["experience"],
        "skills": candidate_skills
    },

    "dataset": {
        "file": "global_ai_jobs_dataset.csv",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_careers": int(len(career_profiles))
    },

    "recommendations": []
}


for _, row in top10.iterrows():

    json_output["recommendations"].append({

        "rank": int(row["rank"]),

        "career": row["career"],

        "career_score": float(
            row["final_score"]
        ),

        "skill_alignment_percentage": float(
            row["skill_alignment"]
        ),

        "matched_skills": (
            []
            if row["matched_skills"] == "None"
            else [
                x.strip()
                for x in row["matched_skills"].split(",")
            ]
        ),

        "skill_gaps": (
            []
            if row["missing_skills"] == "None"
            else [
                x.strip()
                for x in row["missing_skills"].split(",")
            ]
        ),

        "dataset_job_count": int(
            row["job_count"]
        )
    })


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_output,
        f,
        indent=4
    )


# ================================================================
# 12. HUMAN-READABLE REPORT
# ================================================================

with open(
    REPORT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write("CAREERCAST - FINAL CAREER RECOMMENDATION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("CANDIDATE PROFILE\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Education: "
        f"{candidate_profile['education']}\n"
    )

    f.write(
        f"Experience: "
        f"{candidate_profile['experience']}\n"
    )

    f.write(
        "Skills: "
        + ", ".join(candidate_skills)
        + "\n\n"
    )

    f.write("TOP 10 CAREER RECOMMENDATIONS\n")
    f.write("-" * 60 + "\n\n")

    for _, row in top10.iterrows():

        f.write(
            f"{int(row['rank'])}. "
            f"{row['career']}\n"
        )

        f.write(
            f"   Career Score: "
            f"{row['final_score']:.4f}\n"
        )

        f.write(
            f"   Skill Alignment: "
            f"{row['skill_alignment']:.2f}%\n"
        )

        f.write(
            f"   Matched Skills: "
            f"{row['matched_skills']}\n"
        )

        f.write(
            f"   Skill Gaps: "
            f"{row['missing_skills']}\n"
        )

        f.write(
            f"   Job Count: "
            f"{int(row['job_count'])}\n\n"
        )


# ================================================================
# 13. FINAL STATUS
# ================================================================

print("\n" + "=" * 80)
print("7. OUTPUT FILES")
print("=" * 80)

print("\nCSV:")
print(OUTPUT_CSV)

print("\nJSON:")
print(OUTPUT_JSON)

print("\nREPORT:")
print(REPORT_TXT)

print("\n" + "=" * 80)
print("FINAL RECOMMENDATION COMPLETE")
print("=" * 80)

print("\nExisting V4 model: NOT MODIFIED")
print("Existing SBERT model: NOT MODIFIED")
print("Existing XGBoost model: NOT MODIFIED")
print("Original dataset: NOT MODIFIED")
print("New CSV: USED AS CAREER RECOMMENDATION SOURCE")

print("\nCareerCast Milestone 2 final recommendation engine completed.")
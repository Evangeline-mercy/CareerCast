import os
import re
import json
from datetime import datetime

import pandas as pd


# ============================================================
# CAREERCAST MILESTONE 2
# FINAL CAREER RECOMMENDATION ENGINE V2
# ============================================================

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("FINAL CAREER RECOMMENDATION ENGINE V2")
print("=" * 80)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "global_ai_jobs_dataset.csv"
)

OUTPUT_BASE = os.path.join(
    BASE_DIR,
    "v4_output"
)


# ============================================================
# 2. CANDIDATE PROFILE
# ============================================================

candidate = {
    "education": "B.E. Electronics and Communication Engineering",

    "experience": "6 months internship experience",

    "skills": [
        "python",
        "sql",
        "machine learning",
        "pandas",
        "numpy",
        "data analysis",
        "tensorflow"
    ]
}


print("\n" + "=" * 80)
print("1. CANDIDATE PROFILE")
print("=" * 80)

print("Education :", candidate["education"])
print("Experience:", candidate["experience"])

print("\nSkills:")
for skill in candidate["skills"]:
    print(" -", skill)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "=" * 80)
print("2. LOADING NEW CSV DATASET")
print("=" * 80)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("Rows    :", len(df))
print("Columns :", len(df.columns))

print("\nColumns detected:")
for col in df.columns:
    print(" -", col)


# ============================================================
# 4. VALIDATE DATASET
# ============================================================

required_columns = [
    "job_title",
    "industry",
    "experience_level",
    "education_required",
    "ai_domain",
    "skills",
    "tools_used"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"\nMissing required columns: {missing}"
    )

print("\nRequired columns: PASS")


# ============================================================
# 5. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = text.replace("&", " and ")

    text = re.sub(r"[/|;]+", ",", text)

    text = re.sub(r"[^a-z0-9+#.\-, ]+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_skills(text):
    text = normalize_text(text)

    if not text:
        return set()

    parts = re.split(r",", text)

    skills = set()

    for part in parts:
        part = part.strip()

        if part:
            skills.add(part)

    return skills


# ============================================================
# 6. NORMALIZE CANDIDATE SKILLS
# ============================================================

candidate_skills = set(
    normalize_text(skill)
    for skill in candidate["skills"]
)

print("\n" + "=" * 80)
print("3. NORMALIZED CANDIDATE SKILLS")
print("=" * 80)

for skill in sorted(candidate_skills):
    print(" -", skill)


# ============================================================
# 7. BUILD CAREER PROFILES
# ============================================================

print("\n" + "=" * 80)
print("4. BUILDING CAREER PROFILES")
print("=" * 80)


career_profiles = {}

for career, group in df.groupby("job_title"):

    skill_counter = {}

    for _, row in group.iterrows():

        combined = str(row.get("skills", ""))

        skills = extract_skills(combined)

        for skill in skills:

            if skill not in skill_counter:
                skill_counter[skill] = 0

            skill_counter[skill] += 1

    career_profiles[career] = {
        "job_count": len(group),
        "skill_counts": skill_counter
    }


print("Unique careers:", len(career_profiles))


# ============================================================
# 8. CAREER MATCHING
# ============================================================

print("\n" + "=" * 80)
print("5. CALCULATING CAREER MATCHING")
print("=" * 80)


results = []

for career, profile in career_profiles.items():

    skill_counts = profile["skill_counts"]

    if not skill_counts:
        continue

    # --------------------------------------------------------
    # Calculate skill coverage
    # --------------------------------------------------------

    matched_skills = []

    for candidate_skill in candidate_skills:

        if candidate_skill in skill_counts:
            matched_skills.append(candidate_skill)

    skill_alignment = (
        len(matched_skills) /
        len(candidate_skills)
    )

    # --------------------------------------------------------
    # Career-specific skill strength
    # --------------------------------------------------------

    total_skill_frequency = sum(
        skill_counts.values()
    )

    candidate_frequency = sum(
        skill_counts.get(skill, 0)
        for skill in candidate_skills
    )

    if total_skill_frequency > 0:

        frequency_score = (
            candidate_frequency /
            total_skill_frequency
        )

    else:

        frequency_score = 0.0


    # --------------------------------------------------------
    # Normalize frequency score
    # --------------------------------------------------------

    frequency_score = min(
        frequency_score * 10,
        1.0
    )


    # --------------------------------------------------------
    # Domain relevance
    # --------------------------------------------------------

    career_text = " ".join([
        normalize_text(career),
        " ".join(
            normalize_text(x)
            for x in group["industry"].astype(str).tolist()
            if x
        ) if False else "",
    ])

    career_text = normalize_text(career)


    domain_keywords = {
        "machine learning": [
            "machine learning",
            "ml engineer",
            "mlops",
            "data scientist"
        ],

        "python": [
            "engineer",
            "scientist",
            "researcher",
            "analyst",
            "developer"
        ],

        "tensorflow": [
            "ai",
            "machine learning",
            "deep learning",
            "computer vision",
            "nlp"
        ]
    }


    domain_bonus = 0.0

    for skill in matched_skills:

        keywords = domain_keywords.get(
            skill,
            []
        )

        for keyword in keywords:

            if keyword in career_text:
                domain_bonus += 0.03


    domain_bonus = min(
        domain_bonus,
        0.15
    )


    # --------------------------------------------------------
    # Final score
    #
    # 60% skill alignment
    # 25% skill frequency
    # 15% domain relevance
    # --------------------------------------------------------

    final_score = (
        0.60 * skill_alignment
        +
        0.25 * frequency_score
        +
        0.15 * domain_bonus
    )


    # --------------------------------------------------------
    # Skill gaps
    # --------------------------------------------------------

    top_career_skills = sorted(
        skill_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    required_skills = [
        skill
        for skill, count in top_career_skills[:15]
    ]

    skill_gaps = [
        skill
        for skill in required_skills
        if skill not in candidate_skills
    ]


    results.append({

        "career": career,

        "score": round(
            final_score,
            4
        ),

        "skill_alignment": round(
            skill_alignment * 100,
            2
        ),

        "matched_skills": sorted(
            matched_skills
        ),

        "skill_gaps": sorted(
            skill_gaps
        ),

        "job_count": profile[
            "job_count"
        ],

        "frequency_score": round(
            frequency_score,
            4
        ),

        "domain_bonus": round(
            domain_bonus,
            4
        )
    })


# ============================================================
# 9. SORT RESULTS
# ============================================================

results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)


# ============================================================
# 10. TOP 10 RECOMMENDATIONS
# ============================================================

top_results = results[:10]

print("\n" + "=" * 80)
print("6. TOP 10 CAREER RECOMMENDATIONS")
print("=" * 80)


for index, result in enumerate(
    top_results,
    start=1
):

    print(
        f"\n{index}. "
        f"{result['career']}"
    )

    print(
        f"   Career Score    : "
        f"{result['score']:.4f}"
    )

    print(
        f"   Skill Alignment : "
        f"{result['skill_alignment']:.2f}%"
    )

    print(
        "   Matched Skills  : "
        +
        (
            ", ".join(
                result["matched_skills"]
            )
            if result["matched_skills"]
            else "None"
        )
    )

    print(
        "   Skill Gaps      : "
        +
        (
            ", ".join(
                result["skill_gaps"]
            )
            if result["skill_gaps"]
            else "None"
        )
    )

    print(
        f"   Dataset Jobs    : "
        f"{result['job_count']}"
    )


# ============================================================
# 11. CANDIDATE SKILL GAP ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. CANDIDATE SKILL GAP ANALYSIS")
print("=" * 80)


all_dataset_skills = set()

for profile in career_profiles.values():

    all_dataset_skills.update(
        profile["skill_counts"].keys()
    )


candidate_found = sorted(
    candidate_skills &
    all_dataset_skills
)

candidate_missing = sorted(
    candidate_skills -
    all_dataset_skills
)


print("\nCandidate skills:")

for skill in sorted(candidate_skills):

    if skill in all_dataset_skills:

        print(
            f" [FOUND] {skill}"
        )

    else:

        print(
            f" [NOT FOUND] {skill}"
        )


# ============================================================
# 12. GLOBAL SKILL GAPS
# ============================================================

global_skill_counter = {}

for profile in career_profiles.values():

    for skill, count in profile[
        "skill_counts"
    ].items():

        if skill not in global_skill_counter:
            global_skill_counter[skill] = 0

        global_skill_counter[skill] += count


top_global_skills = sorted(
    global_skill_counter.items(),
    key=lambda x: x[1],
    reverse=True
)


recommended_skills = []

for skill, count in top_global_skills:

    if skill not in candidate_skills:

        recommended_skills.append(
            skill
        )

    if len(recommended_skills) >= 15:
        break


print("\nPotential additional skills:")

for skill in recommended_skills:

    print(
        f" [GAP] {skill}"
    )


# ============================================================
# 13. SAVE OUTPUT
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_dir = os.path.join(
    OUTPUT_BASE,
    f"final_v2_{timestamp}"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# CSV

csv_path = os.path.join(
    output_dir,
    "final_recommendations_v2.csv"
)

output_df = pd.DataFrame(
    top_results
)

output_df.insert(
    0,
    "rank",
    range(
        1,
        len(output_df) + 1
    )
)

output_df.to_csv(
    csv_path,
    index=False
)


# JSON

json_path = os.path.join(
    output_dir,
    "final_recommendations_v2.json"
)

json_data = {
    "candidate": candidate,
    "dataset": {
        "rows": len(df),
        "columns": len(df.columns),
        "unique_careers": len(
            career_profiles
        )
    },
    "recommendations": top_results,
    "recommended_skills": recommended_skills
}


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_data,
        f,
        indent=4
    )


# TXT REPORT

report_path = os.path.join(
    output_dir,
    "career_report_v2.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CAREERCAST MILESTONE 2\n"
    )

    f.write(
        "FINAL CAREER RECOMMENDATION ENGINE V2\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Dataset Rows: {len(df)}\n"
    )

    f.write(
        f"Dataset Columns: {len(df.columns)}\n"
    )

    f.write(
        f"Unique Careers: "
        f"{len(career_profiles)}\n\n"
    )

    f.write(
        "CANDIDATE SKILLS\n"
    )

    for skill in sorted(candidate_skills):

        f.write(
            f"- {skill}\n"
        )

    f.write(
        "\nTOP CAREER RECOMMENDATIONS\n"
    )

    for index, result in enumerate(
        top_results,
        start=1
    ):

        f.write(
            f"\n{index}. "
            f"{result['career']}\n"
        )

        f.write(
            f"Score: "
            f"{result['score']:.4f}\n"
        )

        f.write(
            f"Skill Alignment: "
            f"{result['skill_alignment']:.2f}%\n"
        )

        f.write(
            "Matched Skills: "
            +
            ", ".join(
                result["matched_skills"]
            )
            + "\n"
        )

        f.write(
            "Skill Gaps: "
            +
            ", ".join(
                result["skill_gaps"]
            )
            + "\n"
        )

        f.write(
            f"Dataset Jobs: "
            f"{result['job_count']}\n"
        )


# ============================================================
# 14. COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("8. OUTPUT FILES")
print("=" * 80)

print("\nCSV:")
print(csv_path)

print("\nJSON:")
print(json_path)

print("\nREPORT:")
print(report_path)

print("\n" + "=" * 80)
print("FINAL V2 RECOMMENDATION COMPLETE")
print("=" * 80)

print("\nDataset:")
print("NOT MODIFIED")

print("Existing V4 model:")
print("NOT MODIFIED")

print("SBERT model:")
print("NOT MODIFIED")

print("XGBoost model:")
print("NOT MODIFIED")

print("\nCareerCast Milestone 2 V2 completed.")
import os
import re
import json
from datetime import datetime

import pandas as pd
import numpy as np


# ============================================================
# CAREERCAST MILESTONE 2
# FINAL CAREER RECOMMENDATION ENGINE V5
# CAREER-SPECIFIC SKILL SCORING
# ============================================================

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("FINAL CAREER RECOMMENDATION ENGINE V5")
print("CAREER-SPECIFIC SKILL SCORING")
print("=" * 80)


# ============================================================
# 1. CANDIDATE PROFILE
# ============================================================

CANDIDATE_EDUCATION = (
    "B.E. Electronics and Communication Engineering"
)

CANDIDATE_EXPERIENCE = (
    "6 months internship experience"
)

CANDIDATE_SKILLS = [
    "python",
    "sql",
    "machine learning",
    "pandas",
    "numpy",
    "data analysis",
    "tensorflow",
]


print("\n" + "=" * 80)
print("1. CANDIDATE PROFILE")
print("=" * 80)

print("Education :", CANDIDATE_EDUCATION)
print("Experience:", CANDIDATE_EXPERIENCE)

print("\nSkills:")
for skill in CANDIDATE_SKILLS:
    print(" -", skill)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "global_ai_jobs_dataset.csv"
)

OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "v4_output"
)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    f"final_v5_{RUN_ID}"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. SKILL NORMALIZATION
# ============================================================

ALIAS_MAP = {
    # Python ecosystem
    "python programming": "python",
    "python programming language": "python",

    "pandas library": "pandas",
    "pandas python": "pandas",

    "numpy library": "numpy",
    "numpy python": "numpy",

    # ML
    "machine-learning": "machine learning",
    "machinelearning": "machine learning",
    "ml": "machine learning",

    "deep-learning": "deep learning",
    "deeplearning": "deep learning",
    "dl": "deep learning",

    # NLP
    "natural language processing": "nlp",
    "natural-language-processing": "nlp",

    # CV
    "computer-vision": "computer vision",
    "computervision": "computer vision",

    # Data
    "data-analysis": "data analysis",
    "dataanalysis": "data analysis",

    # Frameworks
    "tensorflow framework": "tensorflow",
    "pytorch framework": "pytorch",

    # Cloud
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "microsoft azure": "azure",

    # Databases
    "structured query language": "sql",

    # Kubernetes
    "k8s": "kubernetes",
}


def normalize_skill(skill):
    if skill is None:
        return ""

    skill = str(skill).strip().lower()

    skill = skill.replace("_", " ")
    skill = skill.replace("-", " ")

    skill = re.sub(r"\s+", " ", skill)
    skill = skill.strip()

    if skill in ALIAS_MAP:
        return ALIAS_MAP[skill]

    return skill


def normalize_skill_set(skills):
    result = set()

    for skill in skills:
        normalized = normalize_skill(skill)

        if normalized:
            result.add(normalized)

    return result


candidate_skills = normalize_skill_set(CANDIDATE_SKILLS)


print("\n" + "=" * 80)
print("2. NORMALIZED CANDIDATE SKILLS")
print("=" * 80)

for skill in sorted(candidate_skills):
    print(" -", skill)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n" + "=" * 80)
print("3. LOADING NEW CSV DATASET")
print("=" * 80)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("Rows    :", len(df))
print("Columns :", len(df.columns))

required_columns = [
    "job_title",
    "skills",
    "tools_used"
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required columns: PASS")


# ============================================================
# 5. PARSE SKILLS
# ============================================================

def parse_skill_field(value):
    if pd.isna(value):
        return []

    text = str(value).lower()

    # Handle common separators
    text = text.replace("|", ",")
    text = text.replace(";", ",")
    text = text.replace("/", ",")

    parts = text.split(",")

    result = []

    for part in parts:
        normalized = normalize_skill(part)

        if normalized:
            result.append(normalized)

    return result


print("\n" + "=" * 80)
print("4. NORMALIZING DATASET SKILLS")
print("=" * 80)

df["_normalized_skills"] = df["skills"].apply(parse_skill_field)
df["_normalized_tools"] = df["tools_used"].apply(parse_skill_field)

print("Dataset skill normalization: COMPLETE")


# ============================================================
# 6. BUILD CAREER PROFILES
# ============================================================

print("\n" + "=" * 80)
print("5. BUILDING CAREER-SPECIFIC PROFILES")
print("=" * 80)

career_profiles = {}

career_job_counts = (
    df["job_title"]
    .value_counts()
    .to_dict()
)

for career, group in df.groupby("job_title"):

    skill_counts = {}

    for skills in group["_normalized_skills"]:

        for skill in skills:

            skill_counts[skill] = (
                skill_counts.get(skill, 0) + 1
            )

    # Tools are included with lower importance
    for tools in group["_normalized_tools"]:

        for tool in tools:

            skill_counts[tool] = (
                skill_counts.get(tool, 0) + 0.5
            )

    career_profiles[career] = skill_counts


print("Unique careers:", len(career_profiles))


# ============================================================
# 7. REMOVE VERY COMMON GENERIC SKILLS
# ============================================================

print("\n" + "=" * 80)
print("6. IDENTIFYING CAREER-SPECIFIC SKILLS")
print("=" * 80)

career_count = len(career_profiles)

skill_career_frequency = {}

for career, skill_counts in career_profiles.items():

    for skill in skill_counts:

        skill_career_frequency[skill] = (
            skill_career_frequency.get(skill, 0) + 1
        )


# Skills appearing in almost every career are less useful
generic_threshold = max(
    1,
    int(career_count * 0.8)
)


def specificity_weight(skill):

    frequency = skill_career_frequency.get(skill, 0)

    if frequency >= generic_threshold:
        return 0.55

    if frequency >= career_count * 0.6:
        return 0.75

    if frequency >= career_count * 0.4:
        return 0.90

    if frequency >= career_count * 0.2:
        return 1.10

    return 1.35


print("Career-specific skill weighting: ENABLED")
print("Generic skill threshold:", generic_threshold)


# ============================================================
# 8. CAREER SCORE
# ============================================================

def calculate_career_score(
    career,
    skill_counts,
    candidate_skills
):

    if not skill_counts:
        return {
            "score": 0.0,
            "skill_alignment": 0.0,
            "weighted_alignment": 0.0,
            "matched_skills": [],
            "skill_gaps": [],
        }

    # --------------------------------------------------------
    # Convert frequency into relative importance
    # --------------------------------------------------------

    max_frequency = max(skill_counts.values())

    weighted_total = 0.0
    weighted_matched = 0.0

    matched_skills = []
    skill_gaps = []

    skill_details = []

    for skill, frequency in skill_counts.items():

        relative_frequency = (
            frequency / max_frequency
        )

        specificity = specificity_weight(skill)

        importance = (
            relative_frequency * specificity
        )

        weighted_total += importance

        if skill in candidate_skills:

            weighted_matched += importance

            matched_skills.append(skill)

        else:

            skill_gaps.append(skill)

        skill_details.append(
            (
                skill,
                importance,
                frequency
            )
        )

    # --------------------------------------------------------
    # Basic skill alignment
    # --------------------------------------------------------

    matched_count = len(matched_skills)

    relevant_skill_count = len(skill_counts)

    skill_alignment = (
        matched_count / len(candidate_skills)
        if candidate_skills
        else 0
    )

    # --------------------------------------------------------
    # Weighted alignment
    # --------------------------------------------------------

    if weighted_total > 0:

        weighted_alignment = (
            weighted_matched / weighted_total
        )

    else:

        weighted_alignment = 0.0

    # --------------------------------------------------------
    # Career-specific skills
    # --------------------------------------------------------

    # Select important skills for this particular career.
    # This prevents all careers from having identical gaps.

    sorted_details = sorted(
        skill_details,
        key=lambda x: x[1],
        reverse=True
    )

    important_details = sorted_details[:15]

    important_skills = [
        item[0]
        for item in important_details
    ]

    important_matched = [
        skill
        for skill in important_skills
        if skill in candidate_skills
    ]

    important_coverage = (
        len(important_matched)
        / len(important_skills)
        if important_skills
        else 0
    )

    # --------------------------------------------------------
    # Career-specific gap penalty
    # --------------------------------------------------------

    top_gaps = [
        skill
        for skill in important_skills
        if skill not in candidate_skills
    ][:10]

    gap_penalty = min(
        len(top_gaps) / 10.0,
        1.0
    )

    # --------------------------------------------------------
    # Candidate skill utilization
    # --------------------------------------------------------

    candidate_used = (
        len(
            set(matched_skills)
            & candidate_skills
        )
        / len(candidate_skills)
        if candidate_skills
        else 0
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = (
        0.40 * weighted_alignment
        +
        0.30 * candidate_used
        +
        0.20 * important_coverage
        +
        0.10 * (1.0 - gap_penalty)
    )

    return {
        "score": float(score),
        "skill_alignment": float(skill_alignment),
        "weighted_alignment": float(weighted_alignment),
        "important_coverage": float(important_coverage),
        "matched_skills": sorted(set(matched_skills)),
        "skill_gaps": top_gaps,
    }


# ============================================================
# 9. CALCULATE ALL CAREERS
# ============================================================

print("\n" + "=" * 80)
print("7. CALCULATING CAREER-SPECIFIC MATCHING")
print("=" * 80)

recommendations = []

for career, skill_counts in career_profiles.items():

    result = calculate_career_score(
        career,
        skill_counts,
        candidate_skills
    )

    job_count = career_job_counts.get(
        career,
        0
    )

    recommendations.append({

        "career": career,

        "career_score":
            result["score"],

        "skill_alignment":
            result["skill_alignment"],

        "weighted_skill_alignment":
            result["weighted_alignment"],

        "important_skill_coverage":
            result["important_coverage"],

        "matched_skills":
            result["matched_skills"],

        "skill_gaps":
            result["skill_gaps"],

        "dataset_job_count":
            int(job_count),
    })


# ============================================================
# 10. SORT
# ============================================================

recommendations.sort(
    key=lambda x: (
        x["career_score"],
        x["weighted_skill_alignment"],
        x["important_skill_coverage"],
    ),
    reverse=True
)


# ============================================================
# 11. TOP 10
# ============================================================

top_recommendations = recommendations[:10]

print("\n" + "=" * 80)
print("8. TOP 10 CAREER RECOMMENDATIONS")
print("=" * 80)

for index, item in enumerate(
    top_recommendations,
    start=1
):

    print(f"\n{index}. {item['career']}")

    print(
        f"   Career Score             : "
        f"{item['career_score']:.4f}"
    )

    print(
        f"   Skill Alignment          : "
        f"{item['skill_alignment'] * 100:.2f}%"
    )

    print(
        f"   Weighted Skill Alignment : "
        f"{item['weighted_skill_alignment'] * 100:.2f}%"
    )

    print(
        f"   Important Skill Coverage : "
        f"{item['important_skill_coverage'] * 100:.2f}%"
    )

    print(
        "   Matched Skills           : "
        + (
            ", ".join(item["matched_skills"])
            if item["matched_skills"]
            else "None"
        )
    )

    print(
        "   Skill Gaps               : "
        + (
            ", ".join(item["skill_gaps"])
            if item["skill_gaps"]
            else "None"
        )
    )

    print(
        f"   Dataset Jobs             : "
        f"{item['dataset_job_count']}"
    )


# ============================================================
# 12. CANDIDATE SKILL GAP ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("9. CANDIDATE SKILL GAP ANALYSIS")
print("=" * 80)

# Skills appearing across the dataset
all_dataset_skills = set()

for skill_counts in career_profiles.values():

    all_dataset_skills.update(
        skill_counts.keys()
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


# Aggregate skill frequency
global_skill_frequency = {}

for career, skill_counts in career_profiles.items():

    for skill, count in skill_counts.items():

        global_skill_frequency[skill] = (
            global_skill_frequency.get(skill, 0)
            + count
        )


candidate_gaps = [
    skill
    for skill, count
    in sorted(
        global_skill_frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )
    if skill not in candidate_skills
]


print("\nPotential additional skills:")

for skill in candidate_gaps[:15]:

    print(
        f" [GAP] {skill}"
    )


# ============================================================
# 13. SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "final_recommendations_v5.csv"
)

csv_rows = []

for rank, item in enumerate(
    top_recommendations,
    start=1
):

    csv_rows.append({

        "rank": rank,

        "career":
            item["career"],

        "career_score":
            round(
                item["career_score"],
                6
            ),

        "skill_alignment":
            round(
                item["skill_alignment"] * 100,
                2
            ),

        "weighted_skill_alignment":
            round(
                item["weighted_skill_alignment"] * 100,
                2
            ),

        "important_skill_coverage":
            round(
                item["important_skill_coverage"] * 100,
                2
            ),

        "matched_skills":
            ", ".join(
                item["matched_skills"]
            ),

        "skill_gaps":
            ", ".join(
                item["skill_gaps"]
            ),

        "dataset_job_count":
            item["dataset_job_count"],
    })


output_df = pd.DataFrame(csv_rows)

output_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 14. SAVE JSON
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "final_recommendations_v5.json"
)

json_data = {

    "version": "V5",

    "candidate": {

        "education":
            CANDIDATE_EDUCATION,

        "experience":
            CANDIDATE_EXPERIENCE,

        "skills":
            sorted(candidate_skills),
    },

    "recommendations":
        top_recommendations,

    "candidate_skill_gaps":
        candidate_gaps[:15],
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


# ============================================================
# 15. SAVE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "career_report_v5.txt"
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
        "FINAL CAREER RECOMMENDATION ENGINE V5\n\n"
    )

    f.write(
        "CANDIDATE PROFILE\n"
    )

    f.write(
        f"Education: {CANDIDATE_EDUCATION}\n"
    )

    f.write(
        f"Experience: {CANDIDATE_EXPERIENCE}\n"
    )

    f.write(
        "Skills: "
        + ", ".join(
            sorted(candidate_skills)
        )
        + "\n\n"
    )

    f.write(
        "TOP CAREER RECOMMENDATIONS\n"
    )

    for rank, item in enumerate(
        top_recommendations,
        start=1
    ):

        f.write(
            f"\n{rank}. {item['career']}\n"
        )

        f.write(
            f"Career Score: "
            f"{item['career_score']:.4f}\n"
        )

        f.write(
            f"Skill Alignment: "
            f"{item['skill_alignment'] * 100:.2f}%\n"
        )

        f.write(
            f"Weighted Skill Alignment: "
            f"{item['weighted_skill_alignment'] * 100:.2f}%\n"
        )

        f.write(
            f"Important Skill Coverage: "
            f"{item.get('important_coverage', item.get('important_skill_coverage', 0)) * 100:.2f}%\n"
        )

        f.write(
            "Matched Skills: "
            + ", ".join(
                item["matched_skills"]
            )
            + "\n"
        )

        f.write(
            "Skill Gaps: "
            + ", ".join(
                item["skill_gaps"]
            )
            + "\n"
        )

        f.write(
            f"Dataset Jobs: "
            f"{item['dataset_job_count']}\n"
        )

    f.write(
        "\n\nCANDIDATE SKILL GAPS\n"
    )

    for skill in candidate_gaps[:15]:

        f.write(
            f"- {skill}\n"
        )


# ============================================================
# 16. FINAL
# ============================================================

print("\n" + "=" * 80)
print("10. OUTPUT FILES")
print("=" * 80)

print("\nCSV:")
print(csv_path)

print("\nJSON:")
print(json_path)

print("\nREPORT:")
print(report_path)

print("\n" + "=" * 80)
print("FINAL V5 RECOMMENDATION COMPLETE")
print("=" * 80)

print("\nDataset: NOT MODIFIED")
print("Existing V4 model: NOT MODIFIED")
print("SBERT model: NOT MODIFIED")
print("XGBoost model: NOT MODIFIED")

print(
    "\nCareerCast Milestone 2 V5 completed."
)
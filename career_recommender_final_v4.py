import os
import re
import json
from datetime import datetime

import pandas as pd


# ============================================================
# CAREERCAST MILESTONE 2
# FINAL CAREER RECOMMENDATION ENGINE V4
# CONSISTENT SKILL NORMALIZATION
# ============================================================

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("FINAL CAREER RECOMMENDATION ENGINE V4")
print("CONSISTENT SKILL NORMALIZATION")
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
    "tensorflow"
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

OUTPUT_BASE = os.path.join(
    BASE_DIR,
    "v4_output"
)


# ============================================================
# 3. CONSISTENT SKILL NORMALIZATION
# ============================================================

# Every candidate skill and every dataset skill passes
# through THIS SAME function.

SKILL_ALIASES = {

    # Python
    "python3": "python",
    "python 3": "python",

    # NumPy
    "numpy": "numpy",
    "num py": "numpy",
    "num-py": "numpy",

    # Pandas
    "pandas": "pandas",
    "panda": "pandas",

    # Machine Learning
    "machine learning": "machine learning",
    "machine-learning": "machine learning",
    "ml": "machine learning",

    # Deep Learning
    "deep learning": "deep learning",
    "deep-learning": "deep learning",
    "dl": "deep learning",

    # NLP
    "natural language processing": "nlp",
    "natural-language-processing": "nlp",
    "nlp": "nlp",

    # Computer Vision
    "computer vision": "computer vision",
    "computer-vision": "computer vision",
    "cv": "computer vision",

    # TensorFlow
    "tensorflow": "tensorflow",
    "tensor flow": "tensorflow",

    # PyTorch
    "pytorch": "pytorch",
    "py torch": "pytorch",

    # SQL
    "sql": "sql",
    "structured query language": "sql",

    # Data Analysis
    "data analysis": "data analysis",
    "data analytics": "data analysis",
    "data-analysis": "data analysis",
    "data-analytics": "data analysis",

    # Statistics
    "statistics": "statistics",
    "statistical analysis": "statistics",

    # Cloud
    "amazon web services": "aws",
    "aws": "aws",

    "microsoft azure": "azure",
    "azure": "azure",

    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "gcp": "gcp",

    # Containers / DevOps
    "docker": "docker",
    "kubernetes": "kubernetes",

    # Spark
    "apache spark": "spark",
    "spark": "spark",
}


def normalize_skill(skill):
    """
    Normalize one skill using the SAME alias dictionary
    for candidate and dataset skills.
    """

    if skill is None:
        return None

    skill = str(skill).strip().lower()

    if not skill:
        return None

    # Remove brackets / quotes
    skill = skill.strip("[](){}'\"")

    # Normalize spaces
    skill = re.sub(r"\s+", " ", skill)

    # Direct alias
    if skill in SKILL_ALIASES:
        return SKILL_ALIASES[skill]

    return skill


def parse_skill_string(value):
    """
    Convert a dataset skills/tools field into normalized skills.
    Handles comma, semicolon and pipe separated values.
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    # Normalize common separators
    value = value.replace(";", ",")
    value = value.replace("|", ",")

    # Handle Python-list-like strings
    value = value.replace("[", "")
    value = value.replace("]", "")
    value = value.replace("'", "")
    value = value.replace('"', "")

    parts = value.split(",")

    normalized = set()

    for part in parts:
        skill = normalize_skill(part)

        if skill:
            normalized.add(skill)

    return normalized


# ============================================================
# 4. NORMALIZE CANDIDATE SKILLS
# ============================================================

candidate_skills = set(
    normalize_skill(skill)
    for skill in CANDIDATE_SKILLS
)

candidate_skills.discard(None)

print("\n" + "=" * 80)
print("2. NORMALIZED CANDIDATE SKILLS")
print("=" * 80)

for skill in sorted(candidate_skills):
    print(" -", skill)


# ============================================================
# 5. LOAD DATASET
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

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# 6. BUILD CAREER PROFILES
# ============================================================

print("\n" + "=" * 80)
print("4. BUILDING CAREER PROFILES")
print("=" * 80)

career_profiles = {}

for career, group in df.groupby("job_title"):

    career_skills = set()

    for value in group["skills"].dropna():
        career_skills.update(
            parse_skill_string(value)
        )

    for value in group["tools_used"].dropna():
        career_skills.update(
            parse_skill_string(value)
        )

    career_profiles[career] = career_skills


print("Unique careers:", len(career_profiles))


# ============================================================
# 7. IMPORTANT SKILLS
# ============================================================

# These skills receive more weight because they are central
# to AI/ML careers.

IMPORTANT_SKILLS = {
    "python": 1.5,
    "machine learning": 1.5,
    "deep learning": 1.4,
    "tensorflow": 1.3,
    "pytorch": 1.3,
    "nlp": 1.3,
    "computer vision": 1.3,
    "statistics": 1.2,
    "data analysis": 1.2,
    "sql": 1.1,
    "pandas": 1.0,
    "numpy": 1.0,
    "aws": 1.0,
    "azure": 1.0,
    "gcp": 1.0,
    "docker": 1.0,
    "kubernetes": 1.0,
    "spark": 1.0
}


# ============================================================
# 8. CAREER MATCHING
# ============================================================

print("\n" + "=" * 80)
print("5. CALCULATING CAREER MATCHING")
print("=" * 80)


results = []

for career, career_skills in career_profiles.items():

    # --------------------------------------------------------
    # Exact skill matching
    # --------------------------------------------------------

    matched_skills = (
        candidate_skills.intersection(career_skills)
    )

    # --------------------------------------------------------
    # Skill alignment
    # --------------------------------------------------------

    skill_alignment = (
        len(matched_skills) /
        len(candidate_skills)
        if candidate_skills
        else 0
    )

    # --------------------------------------------------------
    # Weighted skill matching
    # --------------------------------------------------------

    candidate_weight_total = sum(
        IMPORTANT_SKILLS.get(skill, 1.0)
        for skill in candidate_skills
    )

    matched_weight = sum(
        IMPORTANT_SKILLS.get(skill, 1.0)
        for skill in matched_skills
    )

    weighted_alignment = (
        matched_weight /
        candidate_weight_total
        if candidate_weight_total
        else 0
    )

    # --------------------------------------------------------
    # Career-specific important skills
    # --------------------------------------------------------

    career_important = {
        skill
        for skill in career_skills
        if skill in IMPORTANT_SKILLS
    }

    important_matched = (
        candidate_skills.intersection(
            career_important
        )
    )

    important_coverage = (
        len(important_matched) /
        len(career_important)
        if career_important
        else 0
    )

    # --------------------------------------------------------
    # Skill gaps
    # --------------------------------------------------------

    skill_gaps = (
        career_important - candidate_skills
    )

    # Limit gaps to the most relevant ones
    skill_gaps = sorted(
        skill_gaps,
        key=lambda x: IMPORTANT_SKILLS.get(x, 1.0),
        reverse=True
    )[:10]

    # --------------------------------------------------------
    # Dataset job count
    # --------------------------------------------------------

    job_count = len(
        df[df["job_title"] == career]
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    # Skill alignment is the primary signal.
    # Weighted alignment gives more importance to core skills.
    # Important coverage gives career-specific relevance.
    # Job frequency is only a very small supporting factor.

    frequency_factor = min(
        job_count / len(df),
        1.0
    )

    final_score = (
        0.45 * skill_alignment +
        0.35 * weighted_alignment +
        0.15 * important_coverage +
        0.05 * frequency_factor
    )

    results.append({

        "career": career,

        "career_score": round(
            final_score, 4
        ),

        "skill_alignment": round(
            skill_alignment * 100, 2
        ),

        "weighted_skill_alignment": round(
            weighted_alignment * 100, 2
        ),

        "important_skill_coverage": round(
            important_coverage * 100, 2
        ),

        "matched_skills": sorted(
            matched_skills
        ),

        "skill_gaps": skill_gaps,

        "dataset_job_count": job_count
    })


# ============================================================
# 9. SORT RESULTS
# ============================================================

results = sorted(
    results,
    key=lambda x: x["career_score"],
    reverse=True
)


# ============================================================
# 10. TOP 10
# ============================================================

top_results = results[:10]

print("\n" + "=" * 80)
print("6. TOP 10 CAREER RECOMMENDATIONS")
print("=" * 80)

for index, result in enumerate(
    top_results,
    start=1
):

    print(f"\n{index}. {result['career']}")

    print(
        f"   Career Score             : "
        f"{result['career_score']:.4f}"
    )

    print(
        f"   Skill Alignment          : "
        f"{result['skill_alignment']:.2f}%"
    )

    print(
        f"   Weighted Skill Alignment : "
        f"{result['weighted_skill_alignment']:.2f}%"
    )

    print(
        f"   Important Skill Coverage : "
        f"{result['important_skill_coverage']:.2f}%"
    )

    print(
        "   Matched Skills           : "
        + (
            ", ".join(result["matched_skills"])
            if result["matched_skills"]
            else "None"
        )
    )

    print(
        "   Skill Gaps               : "
        + (
            ", ".join(result["skill_gaps"])
            if result["skill_gaps"]
            else "None"
        )
    )

    print(
        f"   Dataset Jobs             : "
        f"{result['dataset_job_count']}"
    )


# ============================================================
# 11. GLOBAL SKILL GAP ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. CANDIDATE SKILL GAP ANALYSIS")
print("=" * 80)

all_dataset_skills = set()

for skills in career_profiles.values():
    all_dataset_skills.update(skills)

print("\nCandidate skills:")

for skill in sorted(candidate_skills):

    if skill in all_dataset_skills:
        print(f" [FOUND] {skill}")
    else:
        print(f" [NOT FOUND] {skill}")


# Skills appearing in the dataset but not candidate
candidate_gaps = (
    all_dataset_skills - candidate_skills
)

# Keep only meaningful AI/ML skills
candidate_gaps = [
    skill
    for skill in candidate_gaps
    if skill in IMPORTANT_SKILLS
]

candidate_gaps = sorted(
    candidate_gaps,
    key=lambda x: IMPORTANT_SKILLS.get(x, 1.0),
    reverse=True
)


print("\nPotential additional skills:")

for skill in candidate_gaps[:15]:
    print(f" [GAP] {skill}")


# ============================================================
# 12. OUTPUT DIRECTORY
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_dir = os.path.join(
    OUTPUT_BASE,
    f"final_v4_{timestamp}"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# 13. CSV OUTPUT
# ============================================================

csv_path = os.path.join(
    output_dir,
    "final_recommendations_v4.csv"
)

csv_rows = []

for rank, result in enumerate(
    top_results,
    start=1
):

    csv_rows.append({

        "rank": rank,

        "career": result["career"],

        "career_score":
            result["career_score"],

        "skill_alignment":
            result["skill_alignment"],

        "weighted_skill_alignment":
            result["weighted_skill_alignment"],

        "important_skill_coverage":
            result["important_skill_coverage"],

        "matched_skills":
            ", ".join(
                result["matched_skills"]
            ),

        "skill_gaps":
            ", ".join(
                result["skill_gaps"]
            ),

        "dataset_job_count":
            result["dataset_job_count"]
    })


output_df = pd.DataFrame(csv_rows)

output_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 14. JSON OUTPUT
# ============================================================

json_path = os.path.join(
    output_dir,
    "final_recommendations_v4.json"
)

json_data = {

    "candidate": {

        "education":
            CANDIDATE_EDUCATION,

        "experience":
            CANDIDATE_EXPERIENCE,

        "skills":
            sorted(candidate_skills)
    },

    "recommendations":
        top_results,

    "global_skill_gaps":
        candidate_gaps[:15],

    "dataset": {

        "rows": len(df),

        "columns": len(df.columns),

        "careers":
            len(career_profiles)
    }
}

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        json_data,
        file,
        indent=4
    )


# ============================================================
# 15. TEXT REPORT
# ============================================================

report_path = os.path.join(
    output_dir,
    "career_report_v4.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CAREERCAST MILESTONE 2\n"
    )

    file.write(
        "FINAL CAREER RECOMMENDATION ENGINE V4\n"
    )

    file.write("=" * 80 + "\n\n")

    file.write(
        "CANDIDATE PROFILE\n"
    )

    file.write(
        f"Education: {CANDIDATE_EDUCATION}\n"
    )

    file.write(
        f"Experience: {CANDIDATE_EXPERIENCE}\n"
    )

    file.write(
        "Skills: "
        + ", ".join(
            sorted(candidate_skills)
        )
        + "\n\n"
    )

    file.write(
        "TOP CAREER RECOMMENDATIONS\n"
    )

    file.write("=" * 80 + "\n")

    for rank, result in enumerate(
        top_results,
        start=1
    ):

        file.write(
            f"\n{rank}. {result['career']}\n"
        )

        file.write(
            f"Career Score: "
            f"{result['career_score']:.4f}\n"
        )

        file.write(
            f"Skill Alignment: "
            f"{result['skill_alignment']:.2f}%\n"
        )

        file.write(
            f"Weighted Skill Alignment: "
            f"{result['weighted_skill_alignment']:.2f}%\n"
        )

        file.write(
            f"Important Skill Coverage: "
            f"{result['important_skill_coverage']:.2f}%\n"
        )

        file.write(
            "Matched Skills: "
            + ", ".join(
                result["matched_skills"]
            )
            + "\n"
        )

        file.write(
            "Skill Gaps: "
            + ", ".join(
                result["skill_gaps"]
            )
            + "\n"
        )

        file.write(
            f"Dataset Jobs: "
            f"{result['dataset_job_count']}\n"
        )

    file.write(
        "\n\nGLOBAL SKILL GAP ANALYSIS\n"
    )

    file.write("=" * 80 + "\n")

    for skill in candidate_gaps[:15]:

        file.write(
            f"- {skill}\n"
        )


# ============================================================
# 16. FINAL STATUS
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
print("FINAL V4 RECOMMENDATION COMPLETE")
print("=" * 80)

print("\nDataset: NOT MODIFIED")
print("Existing V4 model: NOT MODIFIED")
print("SBERT model: NOT MODIFIED")
print("XGBoost model: NOT MODIFIED")

print(
    "\nSkill normalization: "
    "CANDIDATE + DATASET USE SAME ALIAS MAP"
)

print(
    "\nCareerCast Milestone 2 V4 completed."
)
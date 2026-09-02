import os
import re
import json
from datetime import datetime

import pandas as pd


# ============================================================
# CAREERCAST MILESTONE 2
# FINAL CAREER RECOMMENDATION ENGINE V3
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


# ============================================================
# 1. CANDIDATE PROFILE
# ============================================================

CANDIDATE = {
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


# ============================================================
# 2. SKILL NORMALIZATION
# ============================================================

SKILL_ALIASES = {
    "python": "python",
    "python3": "python",

    "sql": "sql",
    "mysql": "sql",
    "postgresql": "sql",

    "pandas": "pandas",
    "pandas library": "pandas",

    "numpy": "numpy",
    "numpy library": "numpy",

    "tensorflow": "tensorflow",
    "tensorflow framework": "tensorflow",

    "pytorch": "pytorch",
    "pytorch framework": "pytorch",

    "machine learning": "machine learning",
    "machine-learning": "machine learning",
    "ml": "machine learning",

    "deep learning": "deep learning",
    "deep-learning": "deep learning",

    "data analysis": "data analysis",
    "data analytics": "data analysis",
    "data analyst": "data analysis",

    "computer vision": "computer vision",
    "computer-vision": "computer vision",

    "natural language processing": "nlp",
    "natural-language processing": "nlp",
    "nlp": "nlp",

    "statistics": "statistics",
    "statistical analysis": "statistics",

    "aws": "aws",
    "amazon web services": "aws",

    "azure": "azure",
    "microsoft azure": "azure",

    "gcp": "gcp",
    "google cloud": "gcp",

    "docker": "docker",
    "kubernetes": "kubernetes",

    "apache spark": "spark",
    "spark": "spark"
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace("&", " and ")

    value = re.sub(r"[\[\]\(\)\{\}]", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_skill(skill):
    skill = normalize_text(skill)

    if skill in SKILL_ALIASES:
        return SKILL_ALIASES[skill]

    return skill


def parse_skill_string(value):
    """
    Convert dataset skill/tool text into normalized skills.
    Handles comma, semicolon, pipe and slash separated values.
    """

    if pd.isna(value):
        return set()

    text = normalize_text(value)

    # Normalize separators
    text = re.sub(r"[;|/]", ",", text)

    pieces = [x.strip() for x in text.split(",")]

    result = set()

    for piece in pieces:

        piece = piece.strip()

        if not piece:
            continue

        normalized = normalize_skill(piece)

        if normalized:
            result.add(normalized)

    return result


# ============================================================
# 3. IMPORTANT SKILL EXTRACTION
# ============================================================

KNOWN_SKILLS = set(SKILL_ALIASES.values())


def extract_known_skills(value):
    """
    Detect known skills even when dataset values contain
    sentences or multiple skill names.
    """

    if pd.isna(value):
        return set()

    text = normalize_text(value)

    found = set()

    # Longest phrases first
    ordered = sorted(
        KNOWN_SKILLS,
        key=len,
        reverse=True
    )

    for skill in ordered:

        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"

        if re.search(pattern, text):

            found.add(skill)

    return found


# ============================================================
# 4. CAREER-SPECIFIC SKILL WEIGHTS
# ============================================================

CAREER_BONUS = {

    "machine learning engineer": {
        "machine learning": 1.35,
        "python": 1.25,
        "tensorflow": 1.15,
        "pytorch": 1.20,
        "deep learning": 1.25,
        "numpy": 1.10,
        "pandas": 1.05
    },

    "ai engineer": {
        "machine learning": 1.30,
        "python": 1.20,
        "tensorflow": 1.15,
        "pytorch": 1.20,
        "deep learning": 1.25
    },

    "ai researcher": {
        "machine learning": 1.30,
        "deep learning": 1.30,
        "pytorch": 1.25,
        "tensorflow": 1.15,
        "statistics": 1.20
    },

    "computer vision engineer": {
        "computer vision": 1.40,
        "deep learning": 1.30,
        "pytorch": 1.25,
        "tensorflow": 1.20
    },

    "data scientist": {
        "python": 1.20,
        "pandas": 1.20,
        "numpy": 1.15,
        "data analysis": 1.25,
        "machine learning": 1.20,
        "statistics": 1.25,
        "sql": 1.10
    },

    "mlops engineer": {
        "machine learning": 1.20,
        "python": 1.15,
        "tensorflow": 1.15,
        "pytorch": 1.15,
        "docker": 1.30,
        "kubernetes": 1.30,
        "aws": 1.20,
        "azure": 1.20,
        "gcp": 1.20
    },

    "nlp engineer": {
        "nlp": 1.40,
        "machine learning": 1.20,
        "deep learning": 1.25,
        "python": 1.20,
        "pytorch": 1.20,
        "tensorflow": 1.15
    },

    "ai product manager": {
        "data analysis": 1.20,
        "machine learning": 1.20,
        "python": 1.05,
        "sql": 1.05
    },

    "data analyst": {
        "data analysis": 1.40,
        "pandas": 1.30,
        "numpy": 1.15,
        "sql": 1.30,
        "python": 1.20,
        "statistics": 1.20
    },

    "prompt engineer": {
        "python": 1.10,
        "nlp": 1.30,
        "machine learning": 1.15,
        "data analysis": 1.10
    }
}


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("FINAL CAREER RECOMMENDATION ENGINE V3")
print("=" * 80)

print("\n" + "=" * 80)
print("1. CANDIDATE PROFILE")
print("=" * 80)

print("Education :", CANDIDATE["education"])
print("Experience:", CANDIDATE["experience"])

print("\nSkills:")

for skill in CANDIDATE["skills"]:
    print(" -", skill)


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
# 6. NORMALIZE CANDIDATE SKILLS
# ============================================================

candidate_skills = {
    normalize_skill(skill)
    for skill in CANDIDATE["skills"]
}

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

career_job_count = {}

career_raw_skills = {}


for _, row in df.iterrows():

    career = normalize_text(row["job_title"])

    if not career:
        continue

    career_job_count[career] = (
        career_job_count.get(career, 0) + 1
    )

    if career not in career_profiles:
        career_profiles[career] = set()

    # Parse both skills and tools_used
    skills_from_column = extract_known_skills(
        row["skills"]
    )

    tools_from_column = extract_known_skills(
        row["tools_used"]
    )

    career_profiles[career].update(
        skills_from_column
    )

    career_profiles[career].update(
        tools_from_column
    )


print(
    "Unique careers:",
    len(career_profiles)
)


# ============================================================
# 8. CALCULATE CAREER MATCHING
# ============================================================

print("\n" + "=" * 80)
print("5. CALCULATING CAREER MATCHING")
print("=" * 80)


results = []


for career, career_skills in career_profiles.items():

    if not career_skills:
        continue

    matched = candidate_skills.intersection(
        career_skills
    )

    missing_skills = career_skills.difference(
        candidate_skills
    )

    # --------------------------------------------------------
    # Basic skill alignment
    # --------------------------------------------------------

    basic_alignment = (
        len(matched) / len(candidate_skills)
        if candidate_skills
        else 0
    )

    # --------------------------------------------------------
    # Career-specific weighted matching
    # --------------------------------------------------------

    bonuses = CAREER_BONUS.get(
        career,
        {}
    )

    weighted_match = 0.0
    weighted_total = 0.0

    for skill in candidate_skills:

        weight = bonuses.get(
            skill,
            1.0
        )

        weighted_total += weight

        if skill in career_skills:
            weighted_match += weight

    weighted_alignment = (
        weighted_match / weighted_total
        if weighted_total
        else 0
    )

    # --------------------------------------------------------
    # Career-specific skill coverage
    # --------------------------------------------------------

    if bonuses:

        important_skills = set(
            bonuses.keys()
        )

        career_important_matched = (
            matched.intersection(
                important_skills
            )
        )

        important_coverage = (
            len(career_important_matched)
            / len(important_skills)
        )

    else:

        important_coverage = basic_alignment


    # --------------------------------------------------------
    # Dataset popularity
    # --------------------------------------------------------

    job_count = career_job_count.get(
        career,
        0
    )

    max_jobs = max(
        career_job_count.values()
    )

    popularity = (
        job_count / max_jobs
        if max_jobs
        else 0
    )


    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    final_score = (
        0.55 * weighted_alignment
        + 0.30 * important_coverage
        + 0.10 * basic_alignment
        + 0.05 * popularity
    )


    # --------------------------------------------------------
    # Skill gaps
    # --------------------------------------------------------

    # Only show useful AI/ML skills as gaps
    useful_gaps = sorted(
        skill
        for skill in missing_skills
        if skill in KNOWN_SKILLS
        and skill not in {
            "python",
            "sql",
            "numpy",
            "pandas"
        }
    )

    results.append({

        "career": career.title(),

        "career_key": career,

        "career_score": round(
            final_score,
            4
        ),

        "skill_alignment": round(
            basic_alignment * 100,
            2
        ),

        "weighted_skill_alignment": round(
            weighted_alignment * 100,
            2
        ),

        "important_skill_coverage": round(
            important_coverage * 100,
            2
        ),

        "matched_skills": sorted(
            matched
        ),

        "skill_gaps": useful_gaps,

        "dataset_job_count": job_count
    })


# ============================================================
# 9. SORT RESULTS
# ============================================================

results = sorted(
    results,
    key=lambda x: (
        x["career_score"],
        x["weighted_skill_alignment"],
        x["dataset_job_count"]
    ),
    reverse=True
)


top_results = results[:10]


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("6. TOP 10 CAREER RECOMMENDATIONS")
print("=" * 80)


for index, result in enumerate(
    top_results,
    start=1
):

    print(
        f"\n{index}. {result['career']}"
    )

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
# 11. CANDIDATE SKILL GAP ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("7. CANDIDATE SKILL GAP ANALYSIS")
print("=" * 80)

dataset_all_skills = set()

for career_skills in career_profiles.values():
    dataset_all_skills.update(
        career_skills
    )


print("\nCandidate skills:")

for skill in sorted(candidate_skills):

    if skill in dataset_all_skills:
        print(
            f" [FOUND] {skill}"
        )
    else:
        print(
            f" [NOT FOUND] {skill}"
        )


# Important:
# Do NOT mark numpy/pandas as missing just because
# a particular career does not require them.

global_gaps = (
    dataset_all_skills
    - candidate_skills
)


# Remove generic / irrelevant skills
global_gaps = {
    skill
    for skill in global_gaps
    if skill in KNOWN_SKILLS
}


print("\nPotential additional skills:")

for skill in sorted(global_gaps):

    print(
        f" [GAP] {skill}"
    )


# ============================================================
# 12. SAVE OUTPUT
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_dir = os.path.join(
    OUTPUT_ROOT,
    f"final_v3_{timestamp}"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

csv_rows = []

for index, result in enumerate(
    top_results,
    start=1
):

    csv_rows.append({

        "rank": index,

        "career": result["career"],

        "career_score": result["career_score"],

        "skill_alignment_percentage":
            result["skill_alignment"],

        "weighted_skill_alignment_percentage":
            result["weighted_skill_alignment"],

        "important_skill_coverage_percentage":
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


output_csv = os.path.join(
    output_dir,
    "final_recommendations_v3.csv"
)

pd.DataFrame(
    csv_rows
).to_csv(
    output_csv,
    index=False
)


# ------------------------------------------------------------
# JSON
# ------------------------------------------------------------

json_output = {

    "project":
        "CareerCast Milestone 2",

    "engine":
        "Final Career Recommendation Engine V3",

    "candidate": {

        "education":
            CANDIDATE["education"],

        "experience":
            CANDIDATE["experience"],

        "skills":
            sorted(candidate_skills)
    },

    "dataset": {

        "file":
            "global_ai_jobs_dataset.csv",

        "rows":
            len(df),

        "columns":
            len(df.columns),

        "unique_careers":
            len(career_profiles)
    },

    "recommendations":
        [
            {
                "rank": i + 1,
                **result
            }

            for i, result
            in enumerate(top_results)
        ],

    "global_skill_gaps":
        sorted(global_gaps)
}


output_json = os.path.join(
    output_dir,
    "final_recommendations_v3.json"
)

with open(
    output_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_output,
        f,
        indent=4
    )


# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

output_report = os.path.join(
    output_dir,
    "career_report_v3.txt"
)


with open(
    output_report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CAREERCAST MILESTONE 2\n"
    )

    f.write(
        "FINAL CAREER RECOMMENDATION ENGINE V3\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Dataset rows: {len(df)}\n"
    )

    f.write(
        f"Dataset columns: {len(df.columns)}\n"
    )

    f.write(
        f"Unique careers: "
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
        "\nTOP RECOMMENDATIONS\n"
    )

    f.write(
        "=" * 80 + "\n"
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
            f"Career Score: "
            f"{result['career_score']:.4f}\n"
        )

        f.write(
            f"Skill Alignment: "
            f"{result['skill_alignment']:.2f}%\n"
        )

        f.write(
            f"Weighted Alignment: "
            f"{result['weighted_skill_alignment']:.2f}%\n"
        )

        f.write(
            "Matched Skills: "
            + ", ".join(
                result["matched_skills"]
            )
            + "\n"
        )

        f.write(
            "Skill Gaps: "
            + ", ".join(
                result["skill_gaps"]
            )
            + "\n"
        )

        f.write(
            f"Dataset Jobs: "
            f"{result['dataset_job_count']}\n"
        )


print("\n" + "=" * 80)
print("8. OUTPUT FILES")
print("=" * 80)

print("\nCSV:")
print(output_csv)

print("\nJSON:")
print(output_json)

print("\nREPORT:")
print(output_report)

print("\n" + "=" * 80)
print("FINAL V3 RECOMMENDATION COMPLETE")
print("=" * 80)

print("\nDataset: NOT MODIFIED")
print("Existing V4 model: NOT MODIFIED")
print("SBERT model: NOT MODIFIED")
print("XGBoost model: NOT MODIFIED")

print("\nCareerCast Milestone 2 V3 completed.")
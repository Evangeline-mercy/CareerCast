import pandas as pd
from collections import Counter, defaultdict

CSV_PATH = "datasets/global_ai_jobs_dataset.csv"

CANDIDATE_SKILLS = {
    "python",
    "sql",
    "machine learning",
    "pandas",
    "numpy",
    "data analysis",
    "tensorflow"
}


def normalize(text):
    if not isinstance(text, str):
        return ""

    return (
        text.lower()
        .strip()
    )


def parse_skills(text):
    if not isinstance(text, str):
        return set()

    return {
        normalize(skill)
        for skill in text.split(";")
        if normalize(skill)
    }


print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("NEW CSV CAREER-SKILL ANALYSIS")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(
    CSV_PATH,
    low_memory=False
)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 1. UNIQUE JOB TITLES
# ============================================================

print("\n" + "=" * 80)
print("1. JOB TITLE ANALYSIS")
print("=" * 80)

job_counts = (
    df["job_title"]
    .value_counts()
)

print(
    "Unique job titles:",
    len(job_counts)
)

print("\nTop job titles:")

print(
    job_counts.head(30).to_string()
)


# ============================================================
# 2. BUILD JOB -> SKILLS
# ============================================================

print("\n" + "=" * 80)
print("2. BUILDING JOB-SKILL PROFILES")
print("=" * 80)

job_skills = defaultdict(Counter)

for _, row in df.iterrows():

    job = normalize(
        row["job_title"]
    )

    skills = parse_skills(
        row["skills"]
    )

    tools = parse_skills(
        row["tools_used"]
    )

    for skill in skills:
        job_skills[job][skill] += 1

    for tool in tools:
        job_skills[job][tool] += 1


print(
    "Job profiles created:",
    len(job_skills)
)


# ============================================================
# 3. DISPLAY JOB PROFILES
# ============================================================

print("\n" + "=" * 80)
print("3. SAMPLE CAREER-SKILL PROFILES")
print("=" * 80)

for job in list(job_skills.keys())[:15]:

    print("\n" + job)

    top_skills = (
        job_skills[job]
        .most_common(10)
    )

    for skill, count in top_skills:

        print(
            f"   {skill}: {count}"
        )


# ============================================================
# 4. CANDIDATE SKILL MATCHING
# ============================================================

print("\n" + "=" * 80)
print("4. CANDIDATE SKILL MATCHING")
print("=" * 80)

career_matches = []

for job, skills_counter in job_skills.items():

    available_skills = set(
        skills_counter.keys()
    )

    matched = (
        CANDIDATE_SKILLS
        &
        available_skills
    )

    score = (
        len(matched)
        /
        len(CANDIDATE_SKILLS)
    )

    career_matches.append(
        (
            job,
            score,
            sorted(matched)
        )
    )


career_matches.sort(
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# 5. TOP MATCHING CAREERS
# ============================================================

print("\n" + "=" * 80)
print("5. TOP CAREERS BASED ON NEW CSV")
print("=" * 80)

for rank, (
    job,
    score,
    matched
) in enumerate(
    career_matches[:30],
    start=1
):

    print(
        f"\n{rank}. {job}"
    )

    print(
        f"   Skill Match: "
        f"{score * 100:.2f}%"
    )

    print(
        "   Matched Skills:",
        ", ".join(matched)
        if matched
        else "None"
    )


# ============================================================
# 6. CANDIDATE SKILL COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("6. CANDIDATE SKILL COVERAGE")
print("=" * 80)

all_dataset_skills = set()

for skills_counter in job_skills.values():

    all_dataset_skills.update(
        skills_counter.keys()
    )


for skill in sorted(
    CANDIDATE_SKILLS
):

    if skill in all_dataset_skills:

        print(
            f"[FOUND] {skill}"
        )

    else:

        print(
            f"[NOT FOUND] {skill}"
        )


print("\n" + "=" * 80)
print("NEW CSV ANALYSIS COMPLETE")
print("=" * 80)
import pandas as pd
import re

DATASET = "datasets/global_ai_jobs_dataset.csv"
VALIDATION = "careercast_curated_validation.csv"

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("V5 CURATED CAREER VALIDATION")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

jobs = pd.read_csv(DATASET)
validation = pd.read_csv(VALIDATION)

print("\nDataset jobs       :", len(jobs))
print("Validation profiles:", len(validation))

# ---------------------------------------------------------
# 2. NORMALIZATION
# ---------------------------------------------------------

ALIASES = {
    "data scientists": "data scientist",
    "data scientist": "data scientist",

    "machine learning engineers": "machine learning engineer",
    "machine learning engineer": "machine learning engineer",

    "data analysts": "data analyst",
    "data analyst": "data analyst",

    "electronics engineers": "electronics engineer",
    "electronics engineer": "electronics engineer",

    "software developers": "software developer",
    "software developer": "software developer",

    "frontend developers": "frontend developer",
    "frontend developer": "frontend developer",

    "embedded systems engineers": "embedded systems engineer",
    "embedded systems engineer": "embedded systems engineer",

    "iot engineers": "iot engineer",
    "iot engineer": "iot engineer",

    "firmware engineers": "firmware engineer",
    "firmware engineer": "firmware engineer",

    "vlsi engineers": "vlsi engineer",
    "vlsi engineer": "vlsi engineer",

    "computer hardware engineers": "computer hardware engineer",
    "computer hardware engineer": "computer hardware engineer",

    "signal processing engineers": "signal processing engineer",
    "signal processing engineer": "signal processing engineer",

    "electrical engineers": "electrical engineer",
    "electrical engineer": "electrical engineer",

    "cloud engineers": "cloud engineer",
    "cloud engineer": "cloud engineer",

    "devops engineers": "devops engineer",
    "devops engineer": "devops engineer",

    "computer and information research scientists":
        "computer and information research scientist",

    "statisticians": "statistician",
    "statistician": "statistician",

    "operations research analysts": "operations research analyst",
    "operations research analyst": "operations research analyst",
}

def normalize(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return ALIASES.get(text, text)

# ---------------------------------------------------------
# 3. CAREER-SPECIFIC SKILL PROFILES
# ---------------------------------------------------------

jobs["career"] = (
    jobs["job_title"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

jobs["skills_clean"] = (
    jobs["skills"]
    .fillna("")
    .astype(str)
    .str.lower()
)

career_profiles = {}

for career in jobs["career"].unique():

    subset = jobs[jobs["career"] == career]

    skills = set()

    for value in subset["skills_clean"]:
        parts = re.split(r"[,;|]", value)

        for skill in parts:
            skill = skill.strip()

            if skill:
                skills.add(skill)

    career_profiles[normalize(career)] = skills

print("\nCareer profiles:", len(career_profiles))

# ---------------------------------------------------------
# 4. SIMPLE V5-STYLE SKILL SCORING
# ---------------------------------------------------------

def parse_skills(text):

    result = set()

    for item in re.split(r"[,;|]", str(text).lower()):

        item = item.strip()

        if item:
            result.add(item)

    return result


def score_candidate(candidate_skills):

    results = []

    for career, career_skills in career_profiles.items():

        if not career_skills:
            continue

        matched = candidate_skills.intersection(career_skills)

        alignment = len(matched) / len(candidate_skills) * 100

        career_coverage = len(matched) / len(career_skills) * 100

        # Balanced recommendation score
        score = (
            0.70 * (alignment / 100)
            + 0.30 * min(career_coverage / 100, 1.0)
        )

        results.append({
            "career": career,
            "score": score,
            "alignment": alignment,
            "matched": sorted(matched)
        })

    return sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

# ---------------------------------------------------------
# 5. VALIDATION
# ---------------------------------------------------------

top1 = 0
top3 = 0
top5 = 0

details = []

for _, row in validation.iterrows():

    candidate_id = row["candidate_id"]

    candidate_skills = parse_skills(row["skills"])

    expected = [
        normalize(x)
        for x in str(row["expected_careers"]).split("|")
    ]

    predictions = score_candidate(candidate_skills)

    top1_predictions = [
        x["career"] for x in predictions[:1]
    ]

    top3_predictions = [
        x["career"] for x in predictions[:3]
    ]

    top5_predictions = [
        x["career"] for x in predictions[:5]
    ]

    hit1 = any(x in expected for x in top1_predictions)
    hit3 = any(x in expected for x in top3_predictions)
    hit5 = any(x in expected for x in top5_predictions)

    if hit1:
        top1 += 1

    if hit3:
        top3 += 1

    if hit5:
        top5 += 1

    details.append({
        "candidate_id": candidate_id,
        "expected": " | ".join(expected),
        "top1": top1_predictions[0] if top1_predictions else "",
        "top3": " | ".join(top3_predictions),
        "top5": " | ".join(top5_predictions),
        "top1_hit": hit1,
        "top3_hit": hit3,
        "top5_hit": hit5
    })

# ---------------------------------------------------------
# 6. RESULTS
# ---------------------------------------------------------

n = len(validation)

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

print(f"\nProfiles evaluated : {n}")

print(
    f"Top-1 Accuracy     : {top1 / n * 100:.2f}%"
)

print(
    f"Top-3 Accuracy     : {top3 / n * 100:.2f}%"
)

print(
    f"Top-5 Accuracy     : {top5 / n * 100:.2f}%"
)

print("\n" + "=" * 80)
print("CANDIDATE RESULTS")
print("=" * 80)

for item in details:

    print(f"\n{item['candidate_id']}")

    print("Expected :", item["expected"])

    print("Top-1    :", item["top1"])

    print("Top-3    :", item["top3"])

    print("Top-5    :", item["top5"])

    print(
        "Hits     :",
        f"Top1={item['top1_hit']},",
        f"Top3={item['top3_hit']},",
        f"Top5={item['top5_hit']}"
    )

# ---------------------------------------------------------
# 7. SAVE
# ---------------------------------------------------------

output = "v4_output/curated_v5_validation.csv"

pd.DataFrame(details).to_csv(
    output,
    index=False
)

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

print("\nSaved:")
print(output)
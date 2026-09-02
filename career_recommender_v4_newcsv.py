import os
import pandas as pd
from collections import Counter
from datetime import datetime

print("=" * 80)
print("CAREERCAST MILESTONE 2")
print("V4 + NEW CSV COMBINED RECOMMENDATION EXPERIMENT")
print("=" * 80)

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Original V4 result - EXACT PATH FOUND FROM DIR COMMAND
OLD_V4_PATH = os.path.join(
    BASE_DIR,
    "v4_output",
    "run_20260821_203417",
    "v4_recommendations.csv"
)

# New CSV
NEW_CSV_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "global_ai_jobs_dataset.csv"
)

# Create separate output folder
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "v4_output",
    "combined_experiment_" + RUN_ID
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

NEW_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "combined_recommendations.csv"
)

COMPARISON_PATH = os.path.join(
    OUTPUT_DIR,
    "old_vs_new_comparison.csv"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

print("\n" + "=" * 80)
print("1. CHECKING INPUT FILES")
print("=" * 80)

print("\nOriginal V4:")
print(OLD_V4_PATH)

if not os.path.exists(OLD_V4_PATH):
    print("\nERROR: Original V4 file was not found.")
    print("Expected:")
    print(OLD_V4_PATH)
    raise SystemExit

print("STATUS: FOUND")

print("\nNew CSV:")
print(NEW_CSV_PATH)

if not os.path.exists(NEW_CSV_PATH):
    print("\nERROR: New CSV was not found.")
    print("Expected:")
    print(NEW_CSV_PATH)
    raise SystemExit

print("STATUS: FOUND")


# ============================================================
# 3. LOAD ORIGINAL V4
# ============================================================

print("\n" + "=" * 80)
print("2. LOADING ORIGINAL V4 TOP-15")
print("=" * 80)

old_v4 = pd.read_csv(OLD_V4_PATH)

print("Rows:", len(old_v4))
print("Columns:", list(old_v4.columns))

print("\nOLD V4 TOP-15:")
print("-" * 80)

for i, row in old_v4.head(15).iterrows():

    career = row.get("Career", row.get("career", "Unknown"))

    score = row.get(
        "Recommendation Score",
        row.get("recommendation_score", "")
    )

    alignment = row.get(
        "Skill Alignment",
        row.get("skill_alignment", "")
    )

    print(
        f"{i + 1}. {career} | "
        f"Score: {score} | "
        f"Skill Alignment: {alignment}"
    )


# ============================================================
# 4. LOAD NEW CSV
# ============================================================

print("\n" + "=" * 80)
print("3. LOADING NEW CSV")
print("=" * 80)

new_df = pd.read_csv(NEW_CSV_PATH)

print("Rows:", len(new_df))
print("Columns:", len(new_df.columns))

print("\nJob title column:", "job_title" in new_df.columns)
print("Skills column:", "skills" in new_df.columns)
print("Tools column:", "tools_used" in new_df.columns)


# ============================================================
# 5. BUILD CAREER-SKILL PROFILES
# ============================================================

print("\n" + "=" * 80)
print("4. BUILDING NEW CSV CAREER-SKILL PROFILES")
print("=" * 80)

career_profiles = {}

for career, group in new_df.groupby("job_title"):

    counter = Counter()

    for skills in group["skills"].dropna():

        skill_list = str(skills).split(";")

        for skill in skill_list:
            skill = skill.strip().lower()

            if skill:
                counter[skill] += 1

    # Also include tools_used
    if "tools_used" in new_df.columns:

        for tools in group["tools_used"].dropna():

            tool_list = str(tools).split(";")

            for tool in tool_list:
                tool = tool.strip().lower()

                if tool:
                    counter[tool] += 1

    career_profiles[career.lower()] = counter


print("Career profiles:", len(career_profiles))


# ============================================================
# 6. CANDIDATE PROFILE
# ============================================================

print("\n" + "=" * 80)
print("5. CANDIDATE PROFILE")
print("=" * 80)

candidate_skills = {
    "python",
    "sql",
    "machine learning",
    "pandas",
    "numpy",
    "data analysis",
    "tensorflow"
}

print("Candidate skills:")

for skill in sorted(candidate_skills):
    print(" -", skill)


# ============================================================
# 7. CALCULATE NEW CSV SKILL MATCH
# ============================================================

print("\n" + "=" * 80)
print("6. CALCULATING NEW CSV CAREER MATCHING")
print("=" * 80)

results = []

for career, skill_counter in career_profiles.items():

    career_skills = set(skill_counter.keys())

    matched = sorted(
        candidate_skills.intersection(career_skills)
    )

    alignment = (
        len(matched) / len(candidate_skills)
        if candidate_skills
        else 0
    )

    # Career-specific skill coverage
    total_skill_frequency = sum(skill_counter.values())

    matched_frequency = sum(
        skill_counter.get(skill, 0)
        for skill in matched
    )

    if total_skill_frequency > 0:
        frequency_score = (
            matched_frequency / total_skill_frequency
        )
    else:
        frequency_score = 0

    # Combined new CSV score
    score = (
        0.7 * alignment +
        0.3 * frequency_score
    )

    results.append({
        "Career": career,
        "New CSV Score": score,
        "Skill Alignment": alignment,
        "Matched Skills": ", ".join(matched) if matched else "None",
        "Matched Skill Count": len(matched)
    })


new_results = pd.DataFrame(results)

new_results = new_results.sort_values(
    by="New CSV Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 8. DISPLAY NEW CSV TOP-15
# ============================================================

print("\n" + "=" * 80)
print("7. TOP-15 CAREERS FROM NEW CSV")
print("=" * 80)

for i, row in new_results.head(15).iterrows():

    print(
        f"\n{i + 1}. {row['Career']}"
    )

    print(
        f"   New CSV Score : "
        f"{row['New CSV Score']:.4f}"
    )

    print(
        f"   Skill Alignment : "
        f"{row['Skill Alignment'] * 100:.2f}%"
    )

    print(
        f"   Matched Skills : "
        f"{row['Matched Skills']}"
    )


# ============================================================
# 9. SAVE NEW CSV RESULTS
# ============================================================

new_results.to_csv(
    NEW_OUTPUT_PATH,
    index=False
)

print("\nNew CSV results saved:")
print(NEW_OUTPUT_PATH)


# ============================================================
# 10. COMPARE OLD V4 VS NEW CSV
# ============================================================

print("\n" + "=" * 80)
print("8. OLD V4 VS NEW CSV COMPARISON")
print("=" * 80)

old_careers = []

for _, row in old_v4.head(15).iterrows():

    career = row.get(
        "Career",
        row.get("career", "Unknown")
    )

    old_careers.append(
        str(career).strip().lower()
    )


new_careers = list(
    new_results.head(15)["Career"]
    .astype(str)
    .str.strip()
    .str.lower()
)


comparison = []

all_careers = []

for career in old_careers:
    if career not in all_careers:
        all_careers.append(career)

for career in new_careers:
    if career not in all_careers:
        all_careers.append(career)


for career in all_careers:

    old_rank = (
        old_careers.index(career) + 1
        if career in old_careers
        else None
    )

    new_rank = (
        new_careers.index(career) + 1
        if career in new_careers
        else None
    )

    if old_rank is not None and new_rank is not None:
        rank_change = old_rank - new_rank
    else:
        rank_change = None

    comparison.append({
        "Career": career,
        "Old V4 Rank": old_rank,
        "New CSV Rank": new_rank,
        "Rank Change": rank_change
    })


comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)


# ============================================================
# 11. PRINT COMPARISON
# ============================================================

print("\nOLD V4 TOP-15")
print("-" * 50)

for i, career in enumerate(old_careers, 1):
    print(f"{i}. {career}")


print("\nNEW CSV TOP-15")
print("-" * 50)

for i, career in enumerate(new_careers, 1):
    print(f"{i}. {career}")


print("\n" + "=" * 80)
print("CAREER MOVEMENT")
print("=" * 80)

for _, row in comparison_df.iterrows():

    if (
        pd.notna(row["Old V4 Rank"])
        and pd.notna(row["New CSV Rank"])
    ):

        print(
            f"{row['Career']}: "
            f"{int(row['Old V4 Rank'])} -> "
            f"{int(row['New CSV Rank'])} "
            f"(Change: {int(row['Rank Change']):+d})"
        )

    elif pd.notna(row["New CSV Rank"]):

        print(
            f"{row['Career']}: "
            f"NEW -> Rank {int(row['New CSV Rank'])}"
        )

    else:

        print(
            f"{row['Career']}: "
            f"OLD -> Rank {int(row['Old V4 Rank'])}"
        )


# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print("\nOriginal V4:")
print("NOT MODIFIED")

print("\nOriginal dataset:")
print("NOT MODIFIED")

print("\nSBERT model:")
print("NOT MODIFIED")

print("\nXGBoost model:")
print("NOT MODIFIED")

print("\nNew CSV:")
print("USED AS ADDITIONAL RECOMMENDATION SOURCE")

print("\nCombined experiment output:")
print(OUTPUT_DIR)

print("\nComparison file:")
print(COMPARISON_PATH)
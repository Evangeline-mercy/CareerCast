import pandas as pd
import joblib
import os
import re


# ============================================================
# CAREERCAST MILESTONE 2
# CAREER RECOMMENDATION + RANKING + SKILL GAP ANALYSIS
# ============================================================

MODEL_DIR = "results/milestone2_profile_model"
PROFILE_FILE = "results/milestone2_unified/career_profiles_96.csv"
VALIDATION_FILE = "careercast_curated_validation.csv"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

OUTPUT_FILE = os.path.join(
    MODEL_DIR,
    "career_recommendations.csv"
)


print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("CAREER RECOMMENDATION + RANKING + SKILL GAP ANALYSIS")
print("=" * 90)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\n[1] Loading trained model...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Model loaded successfully.")
print("Vectorizer loaded successfully.")


# ============================================================
# 2. LOAD CAREER PROFILES
# ============================================================

print("\n[2] Loading career profiles...")

profiles = pd.read_csv(PROFILE_FILE)

print("Career profiles :", len(profiles))
print("Columns         :", list(profiles.columns))


# ============================================================
# 3. LOAD VALIDATION CANDIDATES
# ============================================================

print("\n[3] Loading candidate profiles...")

candidates = pd.read_csv(VALIDATION_FILE)

print("Candidates :", len(candidates))


# ============================================================
# 4. PREPARE CAREER SKILLS
# ============================================================

print("\n[4] Preparing career skill profiles...")


def clean_skill(skill):
    skill = str(skill).strip().lower()
    skill = re.sub(r"\s+", " ", skill)
    return skill


career_skill_map = {}

for _, row in profiles.iterrows():

    career = str(row["job_title"]).strip()

    skills = [
        clean_skill(skill)
        for skill in str(row["required_skills"]).split("|")
        if str(skill).strip()
    ]

    career_skill_map[career] = set(skills)


print(
    "Career skill profiles prepared :",
    len(career_skill_map)
)


# ============================================================
# 5. GENERATE TF-IDF FEATURES
# ============================================================

print("\n[5] Creating candidate TF-IDF representation...")

candidate_skills_text = (
    candidates["skills"]
    .fillna("")
    .astype(str)
    .str.lower()
)

X = vectorizer.transform(candidate_skills_text)

print("TF-IDF shape :", X.shape)


# ============================================================
# 6. PREDICT CAREERS
# ============================================================

print("\n[6] Generating career predictions...")

probabilities = model.predict_proba(X)
classes = model.classes_


# ============================================================
# 7. CAREER RANKING + SKILL GAP
# ============================================================

recommendation_records = []


for candidate_index, candidate in candidates.iterrows():

    candidate_id = candidate["candidate_id"]

    candidate_skills = {
        clean_skill(skill)
        for skill in str(candidate["skills"]).split(",")
        if str(skill).strip()
    }

    # --------------------------------------------------------
    # Get model ranking
    # --------------------------------------------------------

    ranked_indices = probabilities[
        candidate_index
    ].argsort()[::-1]

    # --------------------------------------------------------
    # Evaluate all careers
    # --------------------------------------------------------

    career_results = []

    for model_index in ranked_indices:

        career = classes[model_index]

        probability = float(
            probabilities[candidate_index][model_index]
        )

        required_skills = career_skill_map.get(
            career,
            set()
        )

        matched_skills = (
            candidate_skills &
            required_skills
        )

        missing_skills = (
            required_skills -
            candidate_skills
        )

        if len(required_skills) > 0:

            skill_match_percentage = (
                len(matched_skills)
                /
                len(required_skills)
            ) * 100

        else:

            skill_match_percentage = 0


        # ----------------------------------------------------
        # Combined recommendation score
        # ----------------------------------------------------

        model_score = probability * 100

        final_score = (
            0.70 * model_score
            +
            0.30 * skill_match_percentage
        )


        career_results.append({
            "career": career,
            "model_probability": probability,
            "model_score": model_score,
            "skill_match_percentage":
                skill_match_percentage,
            "matched_skills":
                matched_skills,
            "missing_skills":
                missing_skills,
            "final_score":
                final_score
        })


    # ========================================================
    # SORT BY FINAL SCORE
    # ========================================================

    career_results = sorted(
        career_results,
        key=lambda x: x["final_score"],
        reverse=True
    )


    # ========================================================
    # SAVE TOP 5
    # ========================================================

    for rank, result in enumerate(
        career_results[:5],
        start=1
    ):

        recommendation_records.append({

            "candidate_id":
                candidate_id,

            "candidate_skills":
                candidate["skills"],

            "rank":
                rank,

            "career":
                result["career"],

            "model_probability":
                round(
                    result["model_probability"],
                    6
                ),

            "model_score":
                round(
                    result["model_score"],
                    2
                ),

            "skill_match_percentage":
                round(
                    result["skill_match_percentage"],
                    2
                ),

            "final_score":
                round(
                    result["final_score"],
                    2
                ),

            "matched_skills":
                " | ".join(
                    sorted(
                        result["matched_skills"]
                    )
                ),

            "missing_skills":
                " | ".join(
                    sorted(
                        result["missing_skills"]
                    )
                )
        })


    # ========================================================
    # DISPLAY TOP 3
    # ========================================================

    print("\n" + "-" * 90)

    print("Candidate :", candidate_id)

    print(
        "Skills    :",
        candidate["skills"]
    )

    print("\nTOP 3 CAREER RECOMMENDATIONS")

    for rank, result in enumerate(
        career_results[:3],
        start=1
    ):

        print(
            f"\n#{rank} {result['career']}"
        )

        print(
            f"  Model probability : "
            f"{result['model_probability'] * 100:.2f}%"
        )

        print(
            f"  Skill match       : "
            f"{result['skill_match_percentage']:.2f}%"
        )

        print(
            f"  Final score       : "
            f"{result['final_score']:.2f}"
        )

        print(
            "  Matched skills    :",
            ", ".join(
                sorted(
                    result["matched_skills"]
                )
            )
        )

        print(
            "  Missing skills    :",
            ", ".join(
                sorted(
                    result["missing_skills"]
                )
            )
        )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("SAVING CAREER RECOMMENDATIONS")
print("=" * 90)

recommendations_df = pd.DataFrame(
    recommendation_records
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

recommendations_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nRows saved :", len(recommendations_df))
print("Candidates :", recommendations_df["candidate_id"].nunique())
print("Recommendations/candidate : 5")

print("\nSaved:")
print(os.path.abspath(OUTPUT_FILE))


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("CAREER RECOMMENDATION COMPLETE")
print("=" * 90)

print("\nPipeline:")
print("Candidate Skills")
print("       ?")
print("TF-IDF Representation")
print("       ?")
print("Logistic Regression")
print("       ?")
print("Career Probability")
print("       ?")
print("Skill Matching")
print("       ?")
print("Skill Gap Analysis")
print("       ?")
print("Combined Ranking")
print("       ?")
print("Top-5 Career Recommendations")

print("\n" + "=" * 90)
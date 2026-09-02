# ============================================================
# CAREERCAST MILESTONE 2
# V4 BATCH CURATED-CANDIDATE VALIDATION
#
# IMPORTANT:
# - Does NOT modify career_recommender_v4.py
# - Does NOT modify V4 presentation demo
# - Does NOT modify models
# - Does NOT modify datasets
# - Does NOT overwrite v4_output
#
# Runs the SAME V4 hybrid pipeline independently for
# the 10 curated validation candidates.
# ============================================================

import os
import json
import importlib.util
import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

V4_FILE = "career_recommender_v4.py"

VALIDATION_FILE = "careercast_curated_validation.csv"

RESULT_DIR = "results/curated_v4_validation"

OUTPUT_CSV = os.path.join(
    RESULT_DIR,
    "v4_curated_candidate_results.csv"
)

OUTPUT_JSON = os.path.join(
    RESULT_DIR,
    "v4_curated_validation_summary.json"
)


# ============================================================
# EXPECTED PRESENTATION DEMO
# ============================================================

PRESENTATION_TOP5 = [
    "Python Engineer",
    "AI/ML Engineer",
    "Data Scientist",
    "Embedded Systems Engineer",
    "IoT Engineer"
]


# ============================================================
# LOAD V4 MODULE
# ============================================================

print("\n" + "=" * 75)
print("CAREERCAST MILESTONE 2")
print("V4 CURATED CANDIDATE BATCH VALIDATION")
print("=" * 75)

print("\nREAD-ONLY VALIDATION")
print("V4 presentation demo will NOT be modified.")
print("V4 recommender will NOT be modified.")
print("Models will NOT be modified.")
print("Dataset will NOT be modified.")


spec = importlib.util.spec_from_file_location(
    "career_recommender_v4",
    V4_FILE
)

v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

print("\nV4 module loaded successfully.")


# ============================================================
# LOAD CURATED DATASET
# ============================================================

print("\n" + "=" * 75)
print("1. LOADING CURATED VALIDATION DATASET")
print("=" * 75)

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_FILE}"
    )

validation_df = pd.read_csv(
    VALIDATION_FILE
)

required_columns = [
    "candidate_id",
    "skills",
    "education",
    "expected_careers"
]

missing = [
    c for c in required_columns
    if c not in validation_df.columns
]

if missing:
    raise ValueError(
        f"Missing validation columns: {missing}"
    )

print("[FOUND]")
print("Rows:", len(validation_df))
print("Columns:", list(validation_df.columns))


# ============================================================
# LOAD V4 DATA / ARTIFACTS
# ============================================================

print("\n" + "=" * 75)
print("2. LOADING V4 ARTIFACTS")
print("=" * 75)

df = v4.load_dataset()

career_skill_sets = (
    v4.build_career_skill_sets(df)
)

idf = v4.build_skill_idf(
    career_skill_sets
)

career_embeddings, career_order = (
    v4.load_career_embeddings()
)

v4.verify_career_alignment(
    career_skill_sets,
    career_order,
    career_embeddings
)

sbert_model = v4.load_sbert()


tfidf = joblib.load(
    v4.TFIDF_PATH
)

svd = joblib.load(
    v4.SVD_PATH
)

riasec_scaler = joblib.load(
    v4.RIASEC_SCALER_PATH
)

xgb_model = joblib.load(
    v4.XGBOOST_PATH
)

label_encoder = joblib.load(
    v4.XGBOOST_LABEL_ENCODER_PATH
)

print("\nV4 artifacts loaded successfully.")
print("Careers:", len(career_order))
print("XGBoost features:", xgb_model.n_features_in_)
print("XGBoost classes:", len(label_encoder.classes_))


# ============================================================
# RIASEC
# ============================================================
#
# The curated validation CSV contains:
# candidate_id, skills, education, expected_careers
#
# It does not contain RIASEC scores.
#
# Therefore neutral RIASEC values are used ONLY for this
# validation run. The presentation candidate is untouched.
#
# ============================================================

NEUTRAL_RIASEC = {
    "Realistic": 5,
    "Investigative": 5,
    "Artistic": 5,
    "Social": 5,
    "Enterprising": 5,
    "Conventional": 5
}


# ============================================================
# HELPER: NORMALIZE CAREER LABEL
# ============================================================

def normalize_career(text):

    text = str(text).strip().lower()

    aliases = {
        "data scientists": "data scientist",
        "machine learning engineers": "machine learning engineer",
        "data analysts": "data analyst",
        "statisticians": "statistician",
        "operations research analysts":
            "operations research analyst",

        "vlsi engineers": "vlsi engineer",
        "computer hardware engineers":
            "computer hardware engineer",
        "electronics engineers":
            "electronics engineer",

        "embedded systems engineers":
            "embedded systems engineer",
        "iot engineers": "iot engineer",
        "firmware engineers": "firmware engineer",

        "web developers": "web developer",
        "software developers": "software developer",
        "frontend developers": "frontend developer",

        "machine learning engineers":
            "machine learning engineer",

        "computer and information research scientists":
            "computer and information research scientist",

        "electrical engineers":
            "electrical engineer",

        "signal processing engineers":
            "signal processing engineer",

        "cloud engineers": "cloud engineer",
        "devops engineers": "devops engineer"
    }

    return aliases.get(text, text)


# ============================================================
# HELPER: EXPECTED CAREERS
# ============================================================

def parse_expected(raw):

    return [
        normalize_career(x)
        for x in str(raw).split("|")
        if str(x).strip()
    ]


# ============================================================
# RUN ONE CANDIDATE
# ============================================================

def run_candidate(
    candidate_id,
    skills,
    education
):

    candidate = {

        "skills": str(skills),

        "education": str(education),

        "experience":
            "Curated validation candidate",

        "job_description":
            f"Candidate with skills {skills} "
            f"and education {education}.",

        "riasec":
            dict(NEUTRAL_RIASEC)
    }


    # --------------------------------------------------------
    # Candidate skills
    # --------------------------------------------------------

    candidate_skills = (
        v4.parse_skill_string(
            candidate["skills"]
        )
    )


    # --------------------------------------------------------
    # Skill matching
    # --------------------------------------------------------

    skill_scores = []

    matched_by_career = {}

    for career in career_order:

        career_skills = (
            career_skill_sets.get(
                career,
                set()
            )
        )

        score, matched = (
            v4.calculate_skill_match(
                candidate_skills,
                career_skills,
                idf
            )
        )

        skill_scores.append(score)

        matched_by_career[
            career
        ] = matched


    skill_scores = np.asarray(
        skill_scores,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # SBERT semantic scores
    # --------------------------------------------------------

    semantic_scores = (
        v4.compute_semantic_scores(
            candidate,
            career_embeddings,
            career_order,
            sbert_model
        )
    )


    # --------------------------------------------------------
    # XGBoost 206 features
    # --------------------------------------------------------

    features = (
        v4.build_xgboost_features(
            candidate,
            tfidf,
            svd,
            riasec_scaler
        )
    )


    if features.shape[1] != 206:

        raise ValueError(
            f"Expected 206 features, "
            f"got {features.shape[1]}"
        )


    # --------------------------------------------------------
    # XGBoost scores
    # --------------------------------------------------------

    xgb_scores = (
        v4.compute_xgboost_scores(
            features,
            xgb_model,
            label_encoder,
            career_order
        )
    )


    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    v4.verify_score_alignment(
        career_order,
        semantic_scores,
        skill_scores,
        xgb_scores
    )


    # --------------------------------------------------------
    # V4 HYBRID SCORE
    # --------------------------------------------------------

    (
        final_scores,
        semantic_normalized,
        skill_normalized,
        xgb_normalized
    ) = v4.calculate_final_scores(
        semantic_scores,
        skill_scores,
        xgb_scores
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence_scores = (
        v4.calculate_recommendation_confidence(
            final_scores
        )
    )


    # --------------------------------------------------------
    # RANK ALL CAREERS
    # --------------------------------------------------------

    ranking = np.argsort(
        final_scores
    )[::-1]


    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    top5 = []

    for index in ranking[:5]:

        career = career_order[index]

        matched = (
            matched_by_career.get(
                career,
                []
            )
        )

        top5.append({

            "career": career,

            "final_score":
                round(
                    float(
                        final_scores[index]
                    ),
                    6
                ),

            "confidence":
                round(
                    float(
                        confidence_scores[index]
                    ),
                    2
                ),

            "semantic_score":
                round(
                    float(
                        semantic_scores[index]
                    ),
                    6
                ),

            "skill_match":
                round(
                    float(
                        skill_scores[index]
                    ),
                    6
                ),

            "xgboost_probability":
                round(
                    float(
                        xgb_scores[index]
                    ),
                    8
                ),

            "matched_skills":
                matched,

            "skill_alignment":
                round(
                    float(
                        v4.calculate_skill_alignment(
                            candidate_skills,
                            matched
                        )
                    ),
                    2
                )
        })


    return top5


# ============================================================
# BATCH VALIDATION
# ============================================================

all_results = []

candidate_hits = 0

top3_hits = 0

top5_hits = 0


print("\n" + "=" * 75)
print("3. RUNNING V4 ON CURATED CANDIDATES")
print("=" * 75)


for _, row in validation_df.iterrows():

    candidate_id = str(
        row["candidate_id"]
    )

    print("\n" + "-" * 75)

    print(
        f"CANDIDATE: {candidate_id}"
    )

    print(
        "Skills:",
        row["skills"]
    )

    expected = parse_expected(
        row["expected_careers"]
    )

    top5 = run_candidate(
        candidate_id,
        row["skills"],
        row["education"]
    )

    predicted = [
        normalize_career(
            item["career"]
        )
        for item in top5
    ]

    hit_rank = None

    for rank, career in enumerate(
        predicted,
        start=1
    ):

        if career in expected:

            hit_rank = rank

            break


    if hit_rank == 1:
        candidate_hits += 1

    if hit_rank is not None:
        top5_hits += 1

    if any(
        career in expected
        for career in predicted[:3]
    ):
        top3_hits += 1


    print("\nTOP-5:")

    for rank, item in enumerate(
        top5,
        start=1
    ):

        print(
            f"{rank}. {item['career']} "
            f"| score={item['final_score']:.4f} "
            f"| skill={item['skill_alignment']:.2f}%"
        )


    print(
        "\nExpected:"
    )

    print(
        " | ".join(expected)
    )


    print(
        "\nGround-truth hit rank:",
        hit_rank if hit_rank else "NONE"
    )


    for rank, item in enumerate(
        top5,
        start=1
    ):

        all_results.append({

            "candidate_id":
                candidate_id,

            "rank":
                rank,

            "career":
                item["career"],

            "final_score":
                item["final_score"],

            "confidence_percentage":
                item["confidence"],

            "semantic_score":
                item["semantic_score"],

            "skill_match":
                item["skill_match"],

            "xgboost_probability":
                item["xgboost_probability"],

            "matched_skills":
                json.dumps(
                    item["matched_skills"]
                ),

            "skill_alignment":
                item["skill_alignment"],

            "expected_careers":
                row["expected_careers"],

            "ground_truth_hit_rank":
                hit_rank if hit_rank else ""
        })


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# METRICS
# ============================================================

total_candidates = len(
    validation_df
)

top1_accuracy = (
    candidate_hits /
    total_candidates
)

top3_accuracy = (
    top3_hits /
    total_candidates
)

top5_accuracy = (
    top5_hits /
    total_candidates
)


summary = {

    "validation_type":
        "Curated V4 candidate validation",

    "candidate_count":
        total_candidates,

    "top1_accuracy":
        round(
            top1_accuracy,
            4
        ),

    "top3_accuracy":
        round(
            top3_accuracy,
            4
        ),

    "top5_accuracy":
        round(
            top5_accuracy,
            4
        ),

    "top1_correct":
        candidate_hits,

    "top3_correct":
        top3_hits,

    "top5_correct":
        top5_hits,

    "presentation_demo_top5":
        PRESENTATION_TOP5,

    "presentation_demo_modified":
        False,

    "v4_recommender_modified":
        False,

    "models_modified":
        False,

    "dataset_modified":
        False,

    "validation_file":
        VALIDATION_FILE
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("FINAL CURATED V4 VALIDATION")
print("=" * 75)

print(
    f"\nCandidates validated : "
    f"{total_candidates}"
)

print(
    f"Top-1 Accuracy       : "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-3 Accuracy       : "
    f"{top3_accuracy * 100:.2f}%"
)

print(
    f"Top-5 Accuracy       : "
    f"{top5_accuracy * 100:.2f}%"
)

print("\n" + "=" * 75)
print("PRESENTATION DEMO PROTECTION")
print("=" * 75)

print("1. Python Engineer")
print("2. AI/ML Engineer")
print("3. Data Scientist")
print("4. Embedded Systems Engineer")
print("5. IoT Engineer")

print("\nPresentation demo: UNCHANGED")

print("\n" + "=" * 75)
print("FILES CREATED")
print("=" * 75)

print(
    "CSV:",
    OUTPUT_CSV
)

print(
    "JSON:",
    OUTPUT_JSON
)

print("\nV4 CURATED BATCH VALIDATION: COMPLETE")
print("=" * 75)
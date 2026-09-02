import pandas as pd
import os

# ============================================================
# CAREERCAST MILESTONE 2
# FINAL EVALUATION REPORT GENERATOR
# ============================================================

BASE_DIR = "results"

TRAINING_FILE = os.path.join(
    BASE_DIR,
    "milestone2_training",
    "career_profile_training_dataset.csv"
)

METRICS_FILE = os.path.join(
    BASE_DIR,
    "milestone2_profile_model",
    "logistic_metrics.csv"
)

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "milestone2_profile_model",
    "logistic_predictions.csv"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "milestone2_profile_model",
    "curated_validation_predictions.csv"
)

RECOMMENDATIONS_FILE = os.path.join(
    BASE_DIR,
    "milestone2_profile_model",
    "career_recommendations.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "milestone2_final"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "milestone2_final_evaluation_report.txt"
)


print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("FINAL EVALUATION REPORT")
print("=" * 90)


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("\n[1] Loading training dataset...")

training_df = pd.read_csv(TRAINING_FILE)

training_rows = len(training_df)
training_careers = training_df["career"].nunique()

print("Training rows    :", training_rows)
print("Career classes   :", training_careers)


# ============================================================
# 2. LOAD MODEL METRICS
# ============================================================

print("\n[2] Loading Logistic Regression metrics...")

metrics_df = pd.read_csv(METRICS_FILE)

print("Metrics loaded.")

print("\nMODEL METRICS")
print(metrics_df.to_string(index=False))


# ============================================================
# 3. LOAD EXTERNAL VALIDATION
# ============================================================

print("\n[3] Loading external validation results...")

validation_df = pd.read_csv(VALIDATION_FILE)

validation_rows = len(validation_df)

top1_accuracy = validation_df["top1_match"].mean()
top3_accuracy = validation_df["top3_match"].mean()

print("Validation candidates :", validation_rows)
print("Top-1 accuracy        :", f"{top1_accuracy:.4f}")
print("Top-3 accuracy        :", f"{top3_accuracy:.4f}")


# ============================================================
# 4. LOAD RECOMMENDATIONS
# ============================================================

print("\n[4] Loading career recommendations...")

recommendations_df = pd.read_csv(RECOMMENDATIONS_FILE)

recommendation_rows = len(recommendations_df)

candidate_count = recommendations_df["candidate_id"].nunique()

print("Recommendation rows :", recommendation_rows)
print("Candidates           :", candidate_count)


# ============================================================
# 5. GENERATE RECOMMENDATION SUMMARY
# ============================================================

print("\n[5] Generating recommendation summary...")

recommendation_summary = (
    recommendations_df
    .sort_values(["candidate_id", "rank"])
    .groupby("candidate_id")
)


# ============================================================
# 6. CREATE REPORT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as report:

    report.write("=" * 90 + "\n")
    report.write("CAREERCAST - MILESTONE 2 FINAL EVALUATION REPORT\n")
    report.write("=" * 90 + "\n\n")

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    report.write("1. MILESTONE 2 OBJECTIVE\n")
    report.write("-" * 90 + "\n")

    report.write(
        "Milestone 2 implements a profile-based career prediction and "
        "recommendation pipeline using TF-IDF skill representation and "
        "Logistic Regression classification.\n\n"
    )

    report.write(
        "The system accepts candidate skill profiles, predicts suitable "
        "career categories, calculates skill overlap, identifies missing "
        "skills, and produces ranked career recommendations.\n\n"
    )


    # --------------------------------------------------------
    # TRAINING DATASET
    # --------------------------------------------------------

    report.write("2. TRAINING DATASET\n")
    report.write("-" * 90 + "\n")

    report.write(
        f"Training samples       : {training_rows}\n"
    )

    report.write(
        f"Career classes         : {training_careers}\n"
    )

    report.write(
        f"Samples per career     : "
        f"{training_df['career'].value_counts().min()} - "
        f"{training_df['career'].value_counts().max()}\n"
    )

    report.write(
        f"Empty skill profiles   : "
        f"{training_df['skills'].fillna('').str.strip().eq('').sum()}\n\n"
    )


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    report.write("3. LOGISTIC REGRESSION MODEL PERFORMANCE\n")
    report.write("-" * 90 + "\n")

    for _, row in metrics_df.iterrows():

        if len(row) >= 2:

            metric_name = str(row.iloc[0])
            metric_value = str(row.iloc[1])

            report.write(
                f"{metric_name:<25}: {metric_value}\n"
            )

    report.write("\n")


    # --------------------------------------------------------
    # EXTERNAL VALIDATION
    # --------------------------------------------------------

    report.write("4. EXTERNAL VALIDATION\n")
    report.write("-" * 90 + "\n")

    report.write(
        f"Candidates tested      : {validation_rows}\n"
    )

    report.write(
        f"Top-1 accuracy         : {top1_accuracy:.4f} "
        f"({top1_accuracy * 100:.2f}%)\n"
    )

    report.write(
        f"Top-3 accuracy         : {top3_accuracy:.4f} "
        f"({top3_accuracy * 100:.2f}%)\n\n"
    )


    # --------------------------------------------------------
    # CANDIDATE VALIDATION DETAILS
    # --------------------------------------------------------

    report.write("5. CANDIDATE-WISE VALIDATION SUMMARY\n")
    report.write("-" * 90 + "\n")

    for _, row in validation_df.iterrows():

        candidate_id = row.get("candidate_id", "")
        top1 = row.get("top1_prediction", "")
        top2 = row.get("top2_prediction", "")
        top3 = row.get("top3_prediction", "")

        top1_match = row.get("top1_match", "")
        top3_match = row.get("top3_match", "")

        report.write(
            f"\nCandidate: {candidate_id}\n"
        )

        report.write(
            f"Top-1: {top1}\n"
        )

        report.write(
            f"Top-2: {top2}\n"
        )

        report.write(
            f"Top-3: {top3}\n"
        )

        report.write(
            f"Top-1 Match: {top1_match}\n"
        )

        report.write(
            f"Top-3 Match: {top3_match}\n"
        )


    # --------------------------------------------------------
    # RECOMMENDATION PIPELINE
    # --------------------------------------------------------

    report.write("\n\n")
    report.write("6. CAREER RECOMMENDATION PIPELINE\n")
    report.write("-" * 90 + "\n")

    report.write(
        "Candidate Skills\n"
        "       |\n"
        "       v\n"
        "TF-IDF Representation\n"
        "       |\n"
        "       v\n"
        "Logistic Regression\n"
        "       |\n"
        "       v\n"
        "Career Probability\n"
        "       |\n"
        "       v\n"
        "Skill Matching\n"
        "       |\n"
        "       v\n"
        "Skill Gap Analysis\n"
        "       |\n"
        "       v\n"
        "Combined Ranking\n"
        "       |\n"
        "       v\n"
        "Top-5 Career Recommendations\n\n"
    )


    # --------------------------------------------------------
    # RECOMMENDATION STATISTICS
    # --------------------------------------------------------

    report.write("7. RECOMMENDATION STATISTICS\n")
    report.write("-" * 90 + "\n")

    report.write(
        f"Candidates processed   : {candidate_count}\n"
    )

    report.write(
        f"Recommendations generated: {recommendation_rows}\n"
    )

    report.write(
        "Recommendations per candidate: 5\n\n"
    )


    # --------------------------------------------------------
    # TOP-5 RECOMMENDATIONS
    # --------------------------------------------------------

    report.write("8. TOP-5 CAREER RECOMMENDATIONS\n")
    report.write("-" * 90 + "\n")

    for candidate_id, group in recommendation_summary:

        report.write(
            f"\nCandidate {candidate_id}\n"
        )

        report.write(
            "-" * 50 + "\n"
        )

        for _, row in group.iterrows():

            career = row["career"]
            probability = row["model_probability"]
            skill_match = row["skill_match_percentage"]
            final_score = row["final_score"]

            report.write(
                f"{int(row['rank'])}. {career}\n"
            )

            report.write(
                f"   Model probability : {probability * 100:.2f}%\n"
            )

            report.write(
                f"   Skill match       : {skill_match:.2f}%\n"
            )

            report.write(
                f"   Final score       : {final_score:.2f}\n"
            )


    # --------------------------------------------------------
    # SKILL GAP ANALYSIS
    # --------------------------------------------------------

    report.write("\n\n")
    report.write("9. SKILL GAP ANALYSIS\n")
    report.write("-" * 90 + "\n")

    report.write(
        "The recommendation system compares candidate skills with "
        "the required skills associated with each recommended career.\n\n"
    )

    report.write(
        "Matched skills are reported separately from missing skills, "
        "allowing the system to provide career-specific upskilling "
        "information.\n\n"
    )


    # --------------------------------------------------------
    # VALIDATION INTERPRETATION
    # --------------------------------------------------------

    report.write("10. VALIDATION INTERPRETATION\n")
    report.write("-" * 90 + "\n")

    report.write(
        f"The external validation produced a Top-1 accuracy of "
        f"{top1_accuracy * 100:.2f}% and a Top-3 accuracy of "
        f"{top3_accuracy * 100:.2f}%.\n\n"
    )

    report.write(
        "Top-3 accuracy is particularly useful for a career "
        "recommendation system because multiple career paths can "
        "reasonably match the same candidate profile.\n\n"
    )

    report.write(
        "The external validation results should be considered more "
        "representative of generalization than the internally generated "
        "synthetic profile test set.\n\n"
    )


    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    report.write("11. CURRENT LIMITATIONS\n")
    report.write("-" * 90 + "\n")

    report.write(
        "1. The training profiles are generated through controlled "
        "skill variations from predefined career profiles.\n"
    )

    report.write(
        "2. The internal test accuracy is therefore not equivalent "
        "to real-world deployment accuracy.\n"
    )

    report.write(
        "3. External validation currently contains 10 curated "
        "candidate profiles.\n"
    )

    report.write(
        "4. Some careers have overlapping skill requirements, which "
        "can cause similar careers to appear in the Top-5 results.\n"
    )

    report.write(
        "5. Additional real-world candidate profiles would strengthen "
        "future validation.\n\n"
    )


    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    report.write("12. MILESTONE 2 STATUS\n")
    report.write("-" * 90 + "\n")

    report.write(
        "Career taxonomy expansion                 : COMPLETE\n"
    )

    report.write(
        "Profile-based training dataset            : COMPLETE\n"
    )

    report.write(
        "TF-IDF feature representation             : COMPLETE\n"
    )

    report.write(
        "Logistic Regression baseline              : COMPLETE\n"
    )

    report.write(
        "External validation                       : COMPLETE\n"
    )

    report.write(
        "Career recommendation engine             : COMPLETE\n"
    )

    report.write(
        "Skill matching                            : COMPLETE\n"
    )

    report.write(
        "Skill gap analysis                        : COMPLETE\n"
    )

    report.write(
        "Top-5 recommendation generation           : COMPLETE\n"
    )

    report.write(
        "Final evaluation report                   : COMPLETE\n\n"
    )

    report.write(
        "=" * 90 + "\n"
    )

    report.write(
        "CAREERCAST MILESTONE 2 COMPLETE\n"
    )

    report.write(
        "=" * 90 + "\n"
    )

print("\n" + "=" * 90)
print("FINAL EVALUATION REPORT GENERATED")
print("=" * 90)

print("\nSaved:")
print(os.path.abspath(OUTPUT_FILE))

print("\nMilestone 2 reporting stage complete.")
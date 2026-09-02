from resume_extractor import analyze_resume
from live_milestone2_predictor import predict_resume


# ============================================================
# REAL RESUME -> MILESTONE 2 PREDICTION TEST
# ============================================================

RESUME_PATH = r"C:\Users\ADMIN\Documents\CareerCast_Milestone2_NEW\Ananya_R_Shankar_Synthetic_Test_Resume.docx"


print("=" * 70)
print("CAREERCAST - REAL RESUME MILESTONE 2 PREDICTION TEST")

print("=" * 70)


# ============================================================
# STEP 1: EXTRACT RESUME
# ============================================================

print("\n[1] Extracting resume...")

resume = analyze_resume(RESUME_PATH)

print("File:", resume["file_name"])
print("Text length:", resume["text_length"])
print("Skill count:", resume["skill_count"])


print("\nEXTRACTED SKILLS:")

for i, skill in enumerate(
    resume["skills"],
    start=1
):
    print(f"{i:2d}. {skill}")


# ============================================================
# STEP 2: RUN CLEAN MILESTONE 2 PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("[2] Running verified Milestone 2 prediction...")
print("=" * 70)

result = predict_resume(
    resume["text"],
    top_k=10
)


# ============================================================
# STEP 3: VERIFY EMBEDDING
# ============================================================

print("\nEmbedding dimension:")
print(result["embedding_dimension"])

if result["embedding_dimension"] != 384:
    raise RuntimeError(
        "ERROR: Expected 384-dimensional embedding."
    )


# ============================================================
# STEP 4: DISPLAY MODEL RESULTS
# ============================================================

for model_name, predictions in result["models"].items():

    print("\n" + "=" * 70)
    print(model_name.upper())
    print("=" * 70)

    for item in predictions:

        print(
            f"{item['rank']:2d}. "
            f"{item['career']:<55} "
            f"{item['probability'] * 100:.4f}%"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("REAL RESUME PREDICTION TEST PASSED")
print("=" * 70)
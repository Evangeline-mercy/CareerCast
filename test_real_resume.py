from resume_extractor import analyze_resume


# ============================================================
# REAL RESUME TEST
# ============================================================

resume_path = r"C:\Users\ADMIN\Downloads\Evangeline_Mercy_V_Resume-2.docx"

result = analyze_resume(resume_path)


print("=" * 70)
print("CAREERCAST - REAL RESUME EXTRACTION TEST")
print("=" * 70)

print("\nFILE:")
print(result["file_name"])

print("\nFILE TYPE:")
print(result["file_type"])

print("\nTEXT LENGTH:")
print(result["text_length"])

print("\nSKILL COUNT:")
print(result["skill_count"])

print("\nEXTRACTED SKILLS:")

for i, skill in enumerate(
    result["skills"],
    start=1
):
    print(f"{i:2d}. {skill}")


print("\n" + "=" * 70)
print("FIRST 2000 CHARACTERS OF RESUME")
print("=" * 70)

print(result["text"][:2000])


if not result["text"].strip():
    raise RuntimeError(
        "Resume text extraction failed."
    )


print("\n" + "=" * 70)
print("TEST PASSED.")
print("=" * 70)
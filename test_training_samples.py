import pandas as pd
from live_milestone2_predictor import predict_resume

DATASET = "results/milestone2_training/career_profile_training_dataset.csv"

TEST_CAREERS = [
    "VLSI Engineer",
    "IoT Engineer",
    "React Developer",
    "Database Administrator",
    "Signal Processing Engineer",
    "AWS Solutions Architect",
]

df = pd.read_csv(DATASET)

print("=" * 90)
print("CAREERCAST - TRAINING SAMPLE PREDICTION TEST")
print("=" * 90)

for career in TEST_CAREERS:

    sample = df[df["career"] == career].iloc[0]

    print("\n" + "=" * 90)
    print("EXPECTED CAREER:", career)
    print("-" * 90)
    print("TRAINING SAMPLE:")
    print(sample["skills"])

    result = predict_resume(
        sample["skills"],
        top_k=5
    )

    for model_name, predictions in result["models"].items():

        print("\n" + model_name)
        print("-" * 70)

        for item in predictions:
            print(
                f"{item['rank']}. "
                f"{item['career']:<50} "
                f"{item['probability'] * 100:.2f}%"
            )

        top1 = predictions[0]["career"]

        print(
            "\nTOP-1:",
            top1,
            "|",
            "CORRECT" if top1 == career else "WRONG"
        )

print("\n" + "=" * 90)
print("TEST COMPLETED")
print("=" * 90)
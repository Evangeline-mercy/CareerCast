from live_milestone2_predictor import predict_resume

tests = {
    "C003_VLSI":
        "Verilog, VLSI, RTL Design, Digital Electronics, FPGA",

    "C004_EMBEDDED":
        "Arduino, Embedded C, Microcontrollers, IoT, Embedded Systems",

    "C005_WEB":
        "HTML, CSS, JavaScript, React, Web Development",

    "C006_DATABASE":
        "SQL, Database Management, Database Design, Python",

    "C009_SIGNAL":
        "MATLAB, Signal Processing, Digital Signal Processing, Communication Systems",

    "C010_CLOUD":
        "AWS, Linux, Cloud Computing, Python, Networking"
}

print("=" * 90)
print("CAREERCAST - CONTROLLED MODEL COMPARISON")
print("=" * 90)

results = {}

for name, text in tests.items():
    print(f"\nProcessing {name}...")
    results[name] = predict_resume(text, top_k=5)

for name in tests:

    for model in [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]:

        print("\n" + "=" * 90)
        print(name + " - " + model)
        print("-" * 90)

        predictions = results[name]["models"][model]

        for item in predictions:
            print(
                f"{item['rank']}. "
                f"{item['career']:<50} "
                f"{item['probability'] * 100:.2f}%"
            )

print("\n" + "=" * 90)
print("TEST COMPLETED")
print("=" * 90)
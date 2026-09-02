import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================

BASE = "results/milestone2_sentence_bert_classifier"

LR_PATH = BASE + "/logistic_regression_model.pkl"
RF_PATH = BASE + "/random_forest_model.pkl"
XGB_PATH = BASE + "/xgboost_model.pkl"
LE_PATH = BASE + "/label_encoder.pkl"

# ============================================================
# TEST PROFILE
# ============================================================

resume_text = """
Python, TensorFlow, SQL, Machine Learning, Pandas, NumPy,
Deep Learning
"""

print("=" * 70)
print("CAREERCAST MILESTONE 2 - STANDALONE PREDICTION TEST")
print("=" * 70)

print("\nTEST INPUT:")
print(resume_text.strip())

# ============================================================
# LOAD SBERT
# ============================================================

print("\n[1] Loading Sentence-BERT...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("SBERT model: all-MiniLM-L6-v2")
print("Expected embedding dimension: 384")

# ============================================================
# GENERATE EMBEDDING
# ============================================================

print("\n[2] Generating embedding...")

embedding = model.encode(
    [resume_text],
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("Embedding shape:", embedding.shape)
print("Embedding dtype:", embedding.dtype)
print("Embedding norm:", np.linalg.norm(embedding[0]))

if embedding.shape != (1, 384):
    raise RuntimeError(
        f"WRONG EMBEDDING SHAPE: {embedding.shape}. "
        "Expected (1, 384)."
    )

if not np.isclose(np.linalg.norm(embedding[0]), 1.0, atol=1e-5):
    raise RuntimeError(
        "Embedding is not normalized to unit norm."
    )

# ============================================================
# LOAD MODELS
# ============================================================

print("\n[3] Loading classifiers...")

lr = joblib.load(LR_PATH)
rf = joblib.load(RF_PATH)
xgb = joblib.load(XGB_PATH)
le = joblib.load(LE_PATH)

print("Logistic Regression loaded")
print("Random Forest loaded")
print("XGBoost loaded")
print("Label Encoder loaded")

print("\nNumber of labels:", len(le.classes_))
print("LR classes:", len(lr.classes_))
print("RF classes:", len(rf.classes_))
print("XGB classes:", len(xgb.classes_))

# ============================================================
# VERIFY CLASS CONSISTENCY
# ============================================================

expected_classes = list(range(len(le.classes_)))

if list(lr.classes_) != expected_classes:
    raise RuntimeError("LR class mapping does not match LabelEncoder.")

if list(rf.classes_) != expected_classes:
    raise RuntimeError("RF class mapping does not match LabelEncoder.")

if list(xgb.classes_) != expected_classes:
    raise RuntimeError("XGB class mapping does not match LabelEncoder.")

print("\nClass mapping verification: PASSED")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def show_predictions(model_name, classifier, X, label_encoder, top_k=10):

    probabilities = classifier.predict_proba(X)[0]

    if len(probabilities) != len(label_encoder.classes_):
        raise RuntimeError(
            f"{model_name}: probability count does not match "
            f"LabelEncoder classes."
        )

    indices = np.argsort(probabilities)[::-1][:top_k]

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        f"{'Rank':<6}"
        f"{'Career':<45}"
        f"{'Probability':>15}"
    )

    print("-" * 70)

    for rank, index in enumerate(indices, start=1):

        career = label_encoder.inverse_transform([index])[0]
        probability = probabilities[index]

        print(
            f"{rank:<6}"
            f"{career:<45}"
            f"{probability * 100:>13.4f}%"
        )

    top_index = indices[0]
    top_career = label_encoder.inverse_transform([top_index])[0]
    top_probability = probabilities[top_index]

    print("\nTOP PREDICTION:")
    print("Career:", top_career)
    print("Probability:", f"{top_probability * 100:.4f}%")

    return top_career, top_probability


# ============================================================
# RUN ALL THREE MODELS
# ============================================================

X = embedding

lr_result = show_predictions(
    "LOGISTIC REGRESSION",
    lr,
    X,
    le,
    top_k=10
)

rf_result = show_predictions(
    "RANDOM FOREST",
    rf,
    X,
    le,
    top_k=10
)

xgb_result = show_predictions(
    "XGBOOST",
    xgb,
    X,
    le,
    top_k=10
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "Logistic Regression:",
    lr_result[0],
    f"({lr_result[1] * 100:.4f}%)"
)

print(
    "Random Forest:",
    rf_result[0],
    f"({rf_result[1] * 100:.4f}%)"
)

print(
    "XGBoost:",
    xgb_result[0],
    f"({xgb_result[1] * 100:.4f}%)"
)

print("\nTEST COMPLETED SUCCESSFULLY.")
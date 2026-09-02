import joblib
import numpy as np

# =========================
# LOAD SAVED ARTIFACTS
# =========================
model = joblib.load("results/tree_model/random_forest_model.joblib")
vectorizer = joblib.load("results/tree_features/tfidf_vectorizer.joblib")
svd = joblib.load("results/tree_features/svd.joblib")
riasec_scaler = joblib.load("results/tree_features/riasec_scaler.joblib")
label_encoder = joblib.load("results/tree_model/label_encoder.joblib")

# =========================
# TEST CANDIDATE
# =========================
skills = """
Python, SQL, Machine Learning, Pandas, NumPy,
Data Analysis, TensorFlow
"""

riasec = [
    6,   # Realistic
    9,   # Investigative
    4,   # Artistic
    5,   # Social
    5,   # Enterprising
    8    # Conventional
]

# =========================
# TEXT PROCESSING
# =========================
tfidf_features = vectorizer.transform([skills])

# Reduce 20,000 TF-IDF features to 200
svd_features = svd.transform(tfidf_features)

# Scale RIASEC exactly as during training
riasec_features = riasec_scaler.transform([riasec])

# Combine ? 206 features
final_features = np.hstack([
    svd_features,
    riasec_features
])

print("TF-IDF shape:", tfidf_features.shape)
print("SVD shape:", svd_features.shape)
print("RIASEC shape:", riasec_features.shape)
print("FINAL FEATURES:", final_features.shape)

# =========================
# PREDICTION
# =========================
probabilities = model.predict_proba(final_features)[0]

# Top 5 careers
top_indices = np.argsort(probabilities)[::-1][:5]

print("\n" + "=" * 60)
print("CAREERCAST - CAREER PREDICTION")
print("=" * 60)

for rank, index in enumerate(top_indices, start=1):
    career = label_encoder.inverse_transform([index])[0]
    probability = probabilities[index] * 100

    print(
        f"{rank}. {career:<55} "
        f"{probability:.2f}%"
    )

print("=" * 60)
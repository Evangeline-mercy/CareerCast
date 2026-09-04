"""Reproducible lightweight accuracy gate for GitHub Actions.

This fixture validates that the CareerCast text-classification pipeline can
train, predict, and exceed the declared CI threshold. It is not a claim about
production or real-world model accuracy.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ci_career_profiles.csv"
CI_ACCURACY_THRESHOLD = 0.80


def test_reproducible_accuracy_gate():
    frame = pd.read_csv(FIXTURE_PATH)
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    train_features = vectorizer.fit_transform(train["text"])
    test_features = vectorizer.transform(test["text"])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_features, train["career"])
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test["career"], predictions)

    print(f"CI fixture accuracy: {accuracy:.4f}")
    print(f"Required threshold: {CI_ACCURACY_THRESHOLD:.4f}")
    assert accuracy >= CI_ACCURACY_THRESHOLD

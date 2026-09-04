from streamlit_app.report_builder import build_career_report


def test_pdf_uses_current_prediction_and_gap_data():
    prediction = {
        "top_predictions": [
            {"rank": 1, "career": "Data Scientist", "probability": 0.91, "model": "Logistic Regression"}
        ]
    }
    recommendation = {
        "recommendations": [
            {
                "rank": 1,
                "career": "Data Scientist",
                "ensemble_score": 0.88,
                "lr_probability": 0.91,
                "rf_probability": 0.85,
                "xgb_probability": 0.86,
            }
        ]
    }
    gap_report = {
        "target_career": "Data Scientist",
        "gap_analysis": [
            {
                "alignment_score": 62.5,
                "matched_skills": ["python", "sql"],
                "missing_skills": [
                    {"skill": "statistics", "weight": 0.9, "priority": "High", "suggestion": "Analyse a real dataset."}
                ],
            }
        ],
    }
    model_info = {
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "n_classes": 96,
        "classifiers": ["logistic_regression", "random_forest", "xgboost"],
    }

    pdf = build_career_report("Python and SQL profile", prediction, recommendation, gap_report, model_info)

    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1500

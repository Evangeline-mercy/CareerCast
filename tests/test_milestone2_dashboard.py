from pathlib import Path

import pandas as pd
import pytest

from streamlit_app.milestone2_dashboard import build_model_metrics_frame, load_tsne_projection


def test_model_metrics_frame_uses_verified_values():
    frame = build_model_metrics_frame(
        {"all_models": {"xgboost": {"best_cv_accuracy": 0.91, "test_accuracy": 0.92, "macro_f1": 0.93, "top3_accuracy": 0.98, "top5_accuracy": 1.0}}}
    )
    assert frame.iloc[0]["Model"] == "XGBoost"
    assert frame.iloc[0]["Macro F1"] == pytest.approx(0.93)


def test_model_metrics_frame_accepts_api_list_shape():
    frame = build_model_metrics_frame(
        [{"model": "random_forest", "test_accuracy": 0.88, "macro_f1": 0.87}]
    )
    assert frame.iloc[0]["Model"] == "Random Forest"
    assert frame.iloc[0]["Test accuracy"] == pytest.approx(0.88)


def test_tsne_projection_requires_expected_columns(tmp_path: Path):
    valid = tmp_path / "valid.csv"
    pd.DataFrame({"career": ["Data Scientist"], "tsne_x": [1.0], "tsne_y": [2.0]}).to_csv(valid, index=False)
    assert len(load_tsne_projection(valid)) == 1

    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"career": ["Data Scientist"], "x": [1.0]}).to_csv(invalid, index=False)
    with pytest.raises(ValueError):
        load_tsne_projection(invalid)

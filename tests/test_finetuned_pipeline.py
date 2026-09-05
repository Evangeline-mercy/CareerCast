"""Tests for atomic fine-tuned/legacy pipeline selection."""

from pathlib import Path

import api.main as api_module


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def test_complete_finetuned_pipeline_is_selected(tmp_path):
    classifier_dir = tmp_path / "results/milestone2_finetuned_sbert_classifiers"
    embedding_dir = tmp_path / "results/semantic_embeddings/sbert_finetuned"
    for filename in (
        "logistic_regression_model.pkl",
        "random_forest_model.pkl",
        "xgboost_model.pkl",
        "label_encoder.pkl",
        "finetuned_classifier_summary.json",
    ):
        _write(classifier_dir / filename)
    _write(embedding_dir / "modules.json")
    _write(embedding_dir / "model.safetensors")

    selected = api_module.resolve_pipeline(tmp_path)

    assert selected["name"] == "finetuned_sbert"
    assert selected["fine_tuned"] is True
    assert selected["classifier_dir"] == classifier_dir
    assert selected["embedding_source"] == embedding_dir


def test_incomplete_finetuned_pipeline_falls_back_as_a_whole(tmp_path):
    classifier_dir = tmp_path / "results/milestone2_finetuned_sbert_classifiers"
    _write(classifier_dir / "logistic_regression_model.pkl")

    selected = api_module.resolve_pipeline(tmp_path)

    assert selected["name"] == "pretrained_sbert_legacy"
    assert selected["fine_tuned"] is False
    assert selected["embedding_source"] == "all-MiniLM-L6-v2"
    assert selected["classifier_dir"] == (
        tmp_path / "results/milestone2_sentence_bert_classifier"
    )

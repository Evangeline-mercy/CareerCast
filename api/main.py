"""
CareerCast Milestone 3 - FastAPI REST Service
=============================================
Endpoints:
  POST /predict          - career prediction from skills text
  POST /recommend        - Top-K career recommendations
  POST /gap-report       - skill gap analysis with suggestions
  GET  /health           - health check
  GET  /models/info      - loaded model info

Run from project root:
  uvicorn api.main:app --reload --port 8000
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.services.gap_analysis import SkillGapAnalyzer, parse_skills
from api.auth import require_authenticated_user

# ---------------------------------------------------------------------------
# Paths — anchored to the repository so startup is independent of cwd.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = PROJECT_ROOT / "results/milestone2_training/career_profile_training_dataset.csv"
LEGACY_CLASSIFIER_DIR = PROJECT_ROOT / "results/milestone2_sentence_bert_classifier"
FINETUNED_CLASSIFIER_DIR = PROJECT_ROOT / "results/milestone2_finetuned_sbert_classifiers"
FINETUNED_EMBEDDING_DIR = PROJECT_ROOT / "results/semantic_embeddings/sbert_finetuned"
LEGACY_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def resolve_pipeline(project_root: Path = PROJECT_ROOT) -> dict:
    """Select one complete pipeline; never mix embeddings and classifiers."""
    fine_classifier_dir = project_root / "results/milestone2_finetuned_sbert_classifiers"
    fine_embedding_dir = project_root / "results/semantic_embeddings/sbert_finetuned"
    required = [
        fine_classifier_dir / "logistic_regression_model.pkl",
        fine_classifier_dir / "random_forest_model.pkl",
        fine_classifier_dir / "xgboost_model.pkl",
        fine_classifier_dir / "label_encoder.pkl",
        fine_classifier_dir / "finetuned_classifier_summary.json",
        fine_embedding_dir / "modules.json",
        fine_embedding_dir / "model.safetensors",
    ]
    if all(path.is_file() and path.stat().st_size > 0 for path in required):
        return {
            "name": "finetuned_sbert",
            "embedding_source": fine_embedding_dir,
            "embedding_display_name": "CareerCast fine-tuned all-MiniLM-L6-v2",
            "classifier_dir": fine_classifier_dir,
            "metrics_filename": "finetuned_classifier_summary.json",
            "fine_tuned": True,
        }

    return {
        "name": "pretrained_sbert_legacy",
        "embedding_source": LEGACY_EMBEDDING_MODEL,
        "embedding_display_name": LEGACY_EMBEDDING_MODEL,
        "classifier_dir": project_root / "results/milestone2_sentence_bert_classifier",
        "metrics_filename": "sbert_classifier_summary.json",
        "fine_tuned": False,
    }


_pipeline_info = resolve_pipeline()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CareerCast API",
    description="Milestone 3 - Career Prediction, Recommendation, and Skill Gap REST Service",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (at startup — not per-request)
# ---------------------------------------------------------------------------
_models = {}
_career_skill_profiles = {}
_gap_analyzer = None


@app.on_event("startup")
async def load_models():
    """Load all models once at startup."""
    global _gap_analyzer, _pipeline_info
    import json
    import pandas as pd

    _models.clear()
    _career_skill_profiles.clear()
    _pipeline_info = resolve_pipeline()
    classifier_dir = _pipeline_info["classifier_dir"]
    model_paths = {
        "lr": classifier_dir / "logistic_regression_model.pkl",
        "rf": classifier_dir / "random_forest_model.pkl",
        "xgb": classifier_dir / "xgboost_model.pkl",
        "le": classifier_dir / "label_encoder.pkl",
    }

    try:
        from sentence_transformers import SentenceTransformer
        allow_download = os.getenv("CAREERCAST_ALLOW_MODEL_DOWNLOAD", "0").lower() in {
            "1", "true", "yes"
        }
        embedding_source = _pipeline_info["embedding_source"]
        load_options = {"local_files_only": True}
        if isinstance(embedding_source, str):
            load_options["local_files_only"] = not allow_download
        _models["sbert"] = SentenceTransformer(str(embedding_source), **load_options)
    except Exception as e:
        print(f"WARNING: Could not load SBERT: {e}")
        _models["sbert"] = None

    for key, path in model_paths.items():
        try:
            _models[key] = joblib.load(path)
        except Exception as e:
            print(f"WARNING: Could not load {key} from {path}: {e}")
            _models[key] = None

    # Build career skill profiles from training data for skill gap analysis.
    if TRAINING_DATA.exists():
        try:
            df = pd.read_csv(TRAINING_DATA)
            for _, row in df.iterrows():
                career = str(row["career"]).strip()
                skills_str = str(row["skills"]).strip()
                skill_set = set(
                    s.strip().lower()
                    for s in skills_str.replace("|", ",").split(",")
                    if s.strip()
                )
                if career not in _career_skill_profiles:
                    _career_skill_profiles[career] = set()
                _career_skill_profiles[career].update(skill_set)
        except Exception as e:
            print(f"WARNING: Could not build career profiles: {e}")

    # Load model metrics for /models/info
    metrics_path = classifier_dir / _pipeline_info["metrics_filename"]
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            _models["metrics"] = json.load(f)
    else:
        _models["metrics"] = {}

    try:
        _gap_analyzer = SkillGapAnalyzer()
    except Exception as e:
        print(f"WARNING: Could not initialize skill-gap analyzer: {e}")
        _gap_analyzer = None

    print(f"Startup complete. Pipeline: {_pipeline_info['name']}, "
          f"SBERT: {_models['sbert'] is not None}, "
          f"LR: {_models['lr'] is not None}, RF: {_models['rf'] is not None}, "
          f"XGB: {_models['xgb'] is not None}, "
          f"Career profiles: {len(_career_skill_profiles)}, "
          f"Gap profiles: {_gap_analyzer.career_count if _gap_analyzer else 0}")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    skills_text: str
    top_k: Optional[int] = 5


class CareerPrediction(BaseModel):
    rank: int
    career: str
    probability: float
    model: str


class PredictResponse(BaseModel):
    top_predictions: List[CareerPrediction]
    embedding_model: str
    input_text: str


class RecommendRequest(BaseModel):
    skills_text: str
    top_k: Optional[int] = 10
    ensemble_weights: Optional[dict] = None  # {"lr": 0.4, "rf": 0.3, "xgb": 0.3}


class RecommendedCareer(BaseModel):
    rank: int
    career: str
    ensemble_score: float
    lr_probability: float
    rf_probability: float
    xgb_probability: float


class RecommendResponse(BaseModel):
    recommendations: List[RecommendedCareer]
    weights_used: dict
    top_k: int


class GapReportRequest(BaseModel):
    skills_text: str
    target_career: Optional[str] = None  # if None, use top predicted career
    top_k_careers: Optional[int] = 5


class MissingSkillItem(BaseModel):
    skill: str
    weight: float
    priority: str
    suggestion: str


class SkillGapItem(BaseModel):
    career: str
    profile_source: str
    matched_skills: List[str]
    missing_skills: List[MissingSkillItem]
    alignment_score: float
    priority_summary: dict


class GapReportResponse(BaseModel):
    candidate_skills: List[str]
    target_career: str
    gap_analysis: List[SkillGapItem]
    top_missing_skills: List[MissingSkillItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    if _models.get("sbert") is None:
        raise HTTPException(status_code=503, detail="SBERT model not loaded")
    embedding = _models["sbert"].encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding


def get_topk_from_model(model_key: str, embedding: np.ndarray, top_k: int):
    model = _models.get(model_key)
    le = _models.get("le")
    if model is None or le is None:
        return None, None
    proba = model.predict_proba(embedding)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]
    careers = le.inverse_transform(top_idx)
    return careers, proba[top_idx], proba


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": _pipeline_info["name"],
        "embedding_fine_tuned": _pipeline_info["fine_tuned"],
        "models_loaded": {k: v is not None for k, v in _models.items() if k != "metrics"},
        "career_profiles_loaded": len(_career_skill_profiles),
        "gap_profiles_loaded": _gap_analyzer.career_count if _gap_analyzer else 0,
    }


@app.get("/models/info")
async def models_info():
    return {
        "pipeline": _pipeline_info["name"],
        "embedding_model": _pipeline_info["embedding_display_name"],
        "embedding_fine_tuned": _pipeline_info["fine_tuned"],
        "embedding_dimension": 384,
        "n_classes": len(_models["le"].classes_) if _models.get("le") is not None else 0,
        "classifiers": ["logistic_regression", "random_forest", "xgboost"],
        "metrics_summary": _models.get("metrics", {}),
        "career_profiles_available": len(_career_skill_profiles),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    _user: dict | None = Depends(require_authenticated_user),
):
    """Return one clear Top-K prediction list from the selected best model."""
    if not request.skills_text.strip():
        raise HTTPException(status_code=400, detail="skills_text cannot be empty")
    if request.top_k is None or not 1 <= request.top_k <= 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    embedding = get_embedding(request.skills_text)
    le = _models.get("le")
    if le is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    model = _models.get("lr")
    if model is None:
        raise HTTPException(status_code=503, detail="Selected prediction model not loaded")

    proba = model.predict_proba(embedding)[0]
    top_idx = np.argsort(proba)[::-1][:request.top_k]
    careers = le.inverse_transform(top_idx)
    top_predictions = [
        CareerPrediction(
            rank=rank,
            career=str(career),
            probability=float(probability),
            model="Logistic Regression",
        )
        for rank, (career, probability) in enumerate(
            zip(careers, proba[top_idx]),
            start=1,
        )
    ]

    return PredictResponse(
        top_predictions=top_predictions,
        embedding_model=_pipeline_info["embedding_display_name"],
        input_text=request.skills_text[:200],
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(
    request: RecommendRequest,
    _user: dict | None = Depends(require_authenticated_user),
):
    """Return ensemble Top-K recommendations with per-model breakdown."""
    if not request.skills_text.strip():
        raise HTTPException(status_code=400, detail="skills_text cannot be empty")
    if request.top_k is None or not 1 <= request.top_k <= 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    # Default weights: LR gets highest weight (best test accuracy: 99.82%)
    weights = request.ensemble_weights or {"lr": 0.40, "rf": 0.30, "xgb": 0.30}
    if set(weights).difference({"lr", "rf", "xgb"}):
        raise HTTPException(status_code=400, detail="Weights may contain only lr, rf and xgb")
    if any(not isinstance(value, (int, float)) or value < 0 for value in weights.values()):
        raise HTTPException(status_code=400, detail="Model weights must be non-negative numbers")
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise HTTPException(status_code=400, detail="At least one model weight must be positive")
    weights = {key: weights.get(key, 0) / weight_total for key in ("lr", "rf", "xgb")}

    embedding = get_embedding(request.skills_text)
    le = _models.get("le")
    if le is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    n_classes = len(le.classes_)
    ensemble = np.zeros(n_classes)
    per_model = {}

    for model_key, weight_key in [("lr", "lr"), ("rf", "rf"), ("xgb", "xgb")]:
        model = _models.get(model_key)
        if model is None:
            continue
        proba = model.predict_proba(embedding)[0]
        ensemble += weights.get(weight_key, 0) * proba
        per_model[weight_key] = proba

    top_idx = np.argsort(ensemble)[::-1][:request.top_k]
    careers = le.inverse_transform(top_idx)

    recommendations = []
    for rank, (career, idx) in enumerate(zip(careers, top_idx), start=1):
        recommendations.append(RecommendedCareer(
            rank=rank,
            career=str(career),
            ensemble_score=float(ensemble[idx]),
            lr_probability=float(per_model.get("lr", np.zeros(n_classes))[idx]),
            rf_probability=float(per_model.get("rf", np.zeros(n_classes))[idx]),
            xgb_probability=float(per_model.get("xgb", np.zeros(n_classes))[idx]),
        ))

    return RecommendResponse(
        recommendations=recommendations,
        weights_used=weights,
        top_k=request.top_k,
    )


@app.post("/gap-report", response_model=GapReportResponse)
async def gap_report(
    request: GapReportRequest,
    _user: dict | None = Depends(require_authenticated_user),
):
    """Skill gap analysis with actionable learning suggestions."""
    if not request.skills_text.strip():
        raise HTTPException(status_code=400, detail="skills_text cannot be empty")
    if request.top_k_careers is None or not 1 <= request.top_k_careers <= 10:
        raise HTTPException(status_code=400, detail="top_k_careers must be between 1 and 10")
    if _gap_analyzer is None:
        raise HTTPException(status_code=503, detail="Skill-gap analyzer not loaded")

    candidate_skills = set(parse_skills(request.skills_text))

    # Determine target career(s)
    embedding = get_embedding(request.skills_text)
    le = _models.get("le")
    if le is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    if request.target_career:
        target_careers = [request.target_career]
    else:
        # Use ensemble top-K as targets
        n_classes = len(le.classes_)
        ensemble = np.zeros(n_classes)
        weights = {"lr": 0.40, "rf": 0.30, "xgb": 0.30}
        for model_key, weight_key in [("lr", "lr"), ("rf", "rf"), ("xgb", "xgb")]:
            model = _models.get(model_key)
            if model is None:
                continue
            proba = model.predict_proba(embedding)[0]
            ensemble += weights[weight_key] * proba
        top_idx = np.argsort(ensemble)[::-1][:request.top_k_careers]
        target_careers = [str(c) for c in le.inverse_transform(top_idx)]

    primary_target = target_careers[0]
    all_missing = {}
    gap_items = []

    for career in target_careers:
        try:
            result = _gap_analyzer.analyze(candidate_skills, career)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        gap_items.append(SkillGapItem(**result))
        for item in result["missing_skills"]:
            existing = all_missing.get(item["skill"])
            if existing is None or item["weight"] > existing["weight"]:
                all_missing[item["skill"]] = item

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    top_missing = sorted(
        all_missing.values(),
        key=lambda item: (priority_order[item["priority"]], -item["weight"], item["skill"]),
    )[:10]

    return GapReportResponse(
        candidate_skills=sorted(candidate_skills),
        target_career=primary_target,
        gap_analysis=gap_items,
        top_missing_skills=top_missing,
    )

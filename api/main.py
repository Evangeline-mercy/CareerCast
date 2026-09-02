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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths — all relative to project root (where uvicorn is launched from)
# ---------------------------------------------------------------------------
CLASSIFIER_DIR = Path("results/milestone2_sentence_bert_classifier")
TRAINING_DATA = Path("results/milestone2_training/career_profile_training_dataset.csv")

LR_PATH = CLASSIFIER_DIR / "logistic_regression_model.pkl"
RF_PATH = CLASSIFIER_DIR / "random_forest_model.pkl"
XGB_PATH = CLASSIFIER_DIR / "xgboost_model.pkl"
LE_PATH = CLASSIFIER_DIR / "label_encoder.pkl"
METRICS_PATH = CLASSIFIER_DIR / "sbert_classifier_summary.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

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


@app.on_event("startup")
async def load_models():
    """Load all models once at startup."""
    import json
    import pandas as pd

    try:
        from sentence_transformers import SentenceTransformer
        _models["sbert"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"WARNING: Could not load SBERT: {e}")
        _models["sbert"] = None

    for key, path in [("lr", LR_PATH), ("rf", RF_PATH), ("xgb", XGB_PATH), ("le", LE_PATH)]:
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
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            _models["metrics"] = json.load(f)
    else:
        _models["metrics"] = {}

    print(f"Startup complete. SBERT: {_models['sbert'] is not None}, "
          f"LR: {_models['lr'] is not None}, RF: {_models['rf'] is not None}, "
          f"XGB: {_models['xgb'] is not None}, "
          f"Career profiles: {len(_career_skill_profiles)}")


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


class SkillGapItem(BaseModel):
    career: str
    matched_skills: List[str]
    missing_skills: List[str]
    alignment_score: float
    suggestions: List[str]


class GapReportResponse(BaseModel):
    candidate_skills: List[str]
    target_career: str
    gap_analysis: List[SkillGapItem]
    top_missing_skills: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ACTIONABLE_SUGGESTIONS = {
    "python": "Complete Python for Data Science (Coursera / fast.ai)",
    "machine learning": "Andrew Ng's ML Specialization (Coursera)",
    "deep learning": "fast.ai Practical Deep Learning course",
    "tensorflow": "TensorFlow Developer Certificate (Google)",
    "pytorch": "PyTorch official tutorials (pytorch.org)",
    "sql": "Mode Analytics SQL Tutorial (free)",
    "docker": "Docker Getting Started tutorial (docs.docker.com)",
    "kubernetes": "Kubernetes Fundamentals (CNCF free training)",
    "aws": "AWS Cloud Practitioner certification",
    "azure": "Microsoft Azure Fundamentals (AZ-900)",
    "react": "React official tutorial (react.dev)",
    "node.js": "The Odin Project Node.js track",
    "spark": "Databricks Community Edition (free)",
    "scala": "Scala Exercises (scala-exercises.org)",
    "java": "MOOC.fi Java Programming (free, University of Helsinki)",
    "git": "Git & GitHub Crash Course (freeCodeCamp YouTube)",
    "linux": "The Linux Command Line (free book, linuxcommand.org)",
    "networking": "CompTIA Network+ certification path",
    "cybersecurity": "Google Cybersecurity Certificate (Coursera)",
    "nlp": "Hugging Face NLP Course (free, huggingface.co/learn)",
    "computer vision": "CS231n Stanford (free lecture videos)",
    "data analysis": "Google Data Analytics Certificate (Coursera)",
    "tableau": "Tableau Public free training videos",
    "power bi": "Microsoft Learn: Power BI (free)",
    "devops": "DevOps Foundations (LinkedIn Learning)",
    "mlflow": "MLflow official quickstart (mlflow.org)",
}


def get_suggestion(skill: str) -> str:
    for key, suggestion in ACTIONABLE_SUGGESTIONS.items():
        if key in skill.lower():
            return suggestion
    return f"Search for '{skill}' tutorials on Coursera, Udemy, or freeCodeCamp"


def parse_skills(text: str) -> List[str]:
    """Normalize skills from comma/pipe/semicolon-separated text."""
    text = text.replace("|", ",").replace(";", ",")
    return [s.strip().lower() for s in text.split(",") if s.strip()]


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
        "models_loaded": {k: v is not None for k, v in _models.items() if k != "metrics"},
        "career_profiles_loaded": len(_career_skill_profiles),
    }


@app.get("/models/info")
async def models_info():
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": 384,
        "n_classes": len(_models["le"].classes_) if _models.get("le") is not None else 0,
        "classifiers": ["logistic_regression", "random_forest", "xgboost"],
        "metrics_summary": _models.get("metrics", {}),
        "career_profiles_available": len(_career_skill_profiles),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Return Top-K predictions from all three classifiers independently."""
    if not request.skills_text.strip():
        raise HTTPException(status_code=400, detail="skills_text cannot be empty")

    embedding = get_embedding(request.skills_text)
    le = _models.get("le")
    if le is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    all_predictions = []
    for model_key, model_name in [("lr", "Logistic Regression"),
                                   ("rf", "Random Forest"),
                                   ("xgb", "XGBoost")]:
        model = _models.get(model_key)
        if model is None:
            continue
        proba = model.predict_proba(embedding)[0]
        top_idx = np.argsort(proba)[::-1][:request.top_k]
        careers = le.inverse_transform(top_idx)
        for rank, (career, prob) in enumerate(zip(careers, proba[top_idx]), start=1):
            all_predictions.append(CareerPrediction(
                rank=rank,
                career=str(career),
                probability=float(prob),
                model=model_name,
            ))

    if not all_predictions:
        raise HTTPException(status_code=503, detail="No classifiers available")

    return PredictResponse(
        top_predictions=all_predictions,
        embedding_model=EMBEDDING_MODEL_NAME,
        input_text=request.skills_text[:200],
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Return ensemble Top-K recommendations with per-model breakdown."""
    if not request.skills_text.strip():
        raise HTTPException(status_code=400, detail="skills_text cannot be empty")

    # Default weights: LR gets highest weight (best test accuracy: 99.82%)
    weights = request.ensemble_weights or {"lr": 0.40, "rf": 0.30, "xgb": 0.30}

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
async def gap_report(request: GapReportRequest):
    """Skill gap analysis with actionable learning suggestions."""
    if not request.skills_text.strip():
        raise HTTPException(status_code=400, detail="skills_text cannot be empty")

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
    all_missing = []
    gap_items = []

    for career in target_careers:
        career_skills = _career_skill_profiles.get(career, set())
        if not career_skills:
            # Try case-insensitive match
            for k in _career_skill_profiles:
                if k.lower() == career.lower():
                    career_skills = _career_skill_profiles[k]
                    break

        matched = sorted(candidate_skills & career_skills)
        missing = sorted(career_skills - candidate_skills)[:15]
        all_missing.extend(missing)

        alignment = len(matched) / len(career_skills) * 100 if career_skills else 0.0
        suggestions = [get_suggestion(s) for s in missing[:5]]

        gap_items.append(SkillGapItem(
            career=career,
            matched_skills=matched,
            missing_skills=missing,
            alignment_score=round(alignment, 2),
            suggestions=suggestions,
        ))

    # Global top missing across all target careers
    from collections import Counter
    top_missing = [s for s, _ in Counter(all_missing).most_common(10)]

    return GapReportResponse(
        candidate_skills=sorted(candidate_skills),
        target_career=primary_target,
        gap_analysis=gap_items,
        top_missing_skills=top_missing,
    )
# ============================================================
# CAREERCAST - MILESTONE 2 ANALYTICS DASHBOARD
# ============================================================
#
# Fixed version:
#   - RF/XGB paths updated to milestone2_random_forest / milestone2_xgboost
#   - XGB feature pipeline corrected (TF-IDF -> SVD 100, no RIASEC)
#   - Overview stats read from model metrics, not old CSV datasets
#   - Resume Prediction page added (PDF / DOCX / TXT / paste)
#
# Run from project root:
#   streamlit run dashboard/app_milestone2.py
# ============================================================

import io
import os
import json
import math
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CareerCast - Milestone 2 Analytics",
    page_icon="CC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
V4_OUTPUT = PROJECT_ROOT / "v4_output"

# ------------------------------------------------------------
# Random Forest  (NEW paths)
# ------------------------------------------------------------
RF_MODEL       = RESULTS / "milestone2_random_forest" / "random_forest_model.pkl"
RF_TFIDF       = RESULTS / "milestone2_random_forest" / "tfidf_vectorizer.pkl"
RF_SVD         = RESULTS / "milestone2_random_forest" / "svd_transformer.pkl"
RF_METRICS     = RESULTS / "milestone2_random_forest" / "random_forest_metrics.csv"
RF_PREDICTIONS = RESULTS / "milestone2_random_forest" / "random_forest_predictions.csv"

# ------------------------------------------------------------
# XGBoost  (NEW paths — no RIASEC scaler, no separate label encoder)
# ------------------------------------------------------------
XGB_MODEL       = RESULTS / "milestone2_xgboost" / "xgboost_model.pkl"
XGB_TFIDF       = RESULTS / "milestone2_xgboost" / "tfidf_vectorizer.pkl"
XGB_SVD         = RESULTS / "milestone2_xgboost" / "svd_transformer.pkl"
XGB_METRICS     = RESULTS / "milestone2_xgboost" / "xgboost_metrics.csv"
XGB_PREDICTIONS = RESULTS / "milestone2_xgboost" / "xgboost_predictions.csv"

# ------------------------------------------------------------
# Logistic Regression  (paths were already correct)
# ------------------------------------------------------------
LOGISTIC_MODEL   = RESULTS / "milestone2_profile_model" / "logistic_model.pkl"
LOGISTIC_TFIDF   = RESULTS / "milestone2_profile_model" / "tfidf_vectorizer.pkl"
LOGISTIC_METRICS = RESULTS / "milestone2_profile_model" / "logistic_metrics.csv"
LOGISTIC_PREDICTIONS = RESULTS / "milestone2_profile_model" / "logistic_predictions.csv"
LOGISTIC_RECOMMENDATIONS = RESULTS / "milestone2_profile_model" / "career_recommendations.csv"

# ------------------------------------------------------------
# SBERT
# ------------------------------------------------------------
SBERT_MODEL_DIR       = RESULTS / "semantic_embeddings" / "sbert_finetuned"
SBERT_EMBEDDINGS      = RESULTS / "semantic_embeddings" / "finetuned_career_embeddings.npy"
SBERT_LABELS          = RESULTS / "semantic_embeddings" / "finetuned_career_labels.csv"
SBERT_PROFILE_EMBEDDINGS = RESULTS / "semantic_embeddings" / "profile_embeddings.npy"
SBERT_PROFILE_METADATA   = RESULTS / "semantic_embeddings" / "profile_metadata.csv"
SBERT_COMPARISON         = RESULTS / "semantic_embeddings" / "sbert_old_vs_finetuned_comparison.csv"

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
SEMEVAL_RESULTS      = RESULTS / "milestone2_semeval" / "semeval_sts_results.json"
SEMEVAL_PREDICTIONS  = RESULTS / "milestone2_semeval" / "semeval_sts_track5_predictions.csv"
LINKEDIN_SUMMARY     = RESULTS / "milestone2_linkedin" / "linkedin_transition_validation_summary.json"
LINKEDIN_RECOMMENDATIONS = RESULTS / "milestone2_linkedin" / "linkedin_transition_recommendations.csv"
HYBRID_SUMMARY       = RESULTS / "hybrid_recommender" / "hybrid_evaluation_summary.json"
ALPHA_RESULTS        = RESULTS / "hybrid_recommender" / "alpha_tuning_results.csv"
CURATED_SUMMARY      = RESULTS / "curated_v4_validation" / "v4_curated_validation_summary.json"
CURATED_RESULTS      = RESULTS / "curated_v4_validation" / "v4_curated_candidate_results.csv"

# ============================================================
# SAFE FILE READERS
# ============================================================

def read_csv_safe(path):
    if not Path(path).exists():
        return None
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    try:
        return pd.read_csv(path, encoding="latin1", encoding_errors="replace", low_memory=False)
    except Exception:
        return None


def read_text_safe(path):
    if not Path(path).exists():
        return ""
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return Path(path).read_text(encoding=enc)
        except Exception:
            pass
    return ""


def read_json_safe(path):
    text = read_text_safe(path)
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_joblib(path_string):
    p = Path(path_string)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


@st.cache_data
def load_npy(path_string):
    p = Path(path_string)
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception:
        return None


# Load all model objects once
RF_MODEL_OBJ    = load_joblib(str(RF_MODEL))
RF_TFIDF_OBJ    = load_joblib(str(RF_TFIDF))
RF_SVD_OBJ      = load_joblib(str(RF_SVD))

XGB_MODEL_OBJ   = load_joblib(str(XGB_MODEL))
XGB_TFIDF_OBJ   = load_joblib(str(XGB_TFIDF))
XGB_SVD_OBJ     = load_joblib(str(XGB_SVD))

LOGISTIC_MODEL_OBJ  = load_joblib(str(LOGISTIC_MODEL))
LOGISTIC_TFIDF_OBJ  = load_joblib(str(LOGISTIC_TFIDF))

CAREER_EMBEDDINGS   = load_npy(str(SBERT_EMBEDDINGS))
PROFILE_EMBEDDINGS  = load_npy(str(SBERT_PROFILE_EMBEDDINGS))

# Build XGB career-name mapping from predictions CSV
# (new XGBoost classes_ are int IDs; career names come from predictions)
@st.cache_data
def build_xgb_career_names():
    """Return ordered list of 96 career names matching XGBoost class 0..95."""
    # Prefer to read from RF model (same dataset, same label order)
    rf = load_joblib(str(RF_MODEL))
    if rf is not None and hasattr(rf, "classes_"):
        classes = rf.classes_
        if len(classes) == 96:
            return list(classes)
    # Fallback: sorted unique careers from predictions CSV
    df = read_csv_safe(XGB_PREDICTIONS)
    if df is not None and "true_career" in df.columns:
        return sorted(df["true_career"].dropna().unique().tolist())
    return []

XGB_CAREER_NAMES = build_xgb_career_names()

# SBERT
SBERT_MODEL_OBJ = None
try:
    from sentence_transformers import SentenceTransformer
    if SBERT_MODEL_DIR.exists():
        SBERT_MODEL_OBJ = SentenceTransformer(str(SBERT_MODEL_DIR))
except Exception:
    SBERT_MODEL_OBJ = None

# ============================================================
# LOAD STATIC DATA
# ============================================================

@st.cache_data
def load_all_data():
    data = {}
    data["rf_metrics"]      = read_csv_safe(RF_METRICS)
    data["rf_predictions"]  = read_csv_safe(RF_PREDICTIONS)
    data["xgb_metrics"]     = read_csv_safe(XGB_METRICS)
    data["xgb_predictions"] = read_csv_safe(XGB_PREDICTIONS)
    data["logistic_metrics"]     = read_csv_safe(LOGISTIC_METRICS)
    data["logistic_predictions"] = read_csv_safe(LOGISTIC_PREDICTIONS)
    data["sbert_comparison"]     = read_csv_safe(SBERT_COMPARISON)
    data["alpha_results"]        = read_csv_safe(ALPHA_RESULTS)
    data["semeval_predictions"]  = read_csv_safe(SEMEVAL_PREDICTIONS)
    data["linkedin_recommendations"] = read_csv_safe(LINKEDIN_RECOMMENDATIONS)
    data["curated_results"]      = read_csv_safe(CURATED_RESULTS)
    data["hybrid_summary"]  = read_json_safe(HYBRID_SUMMARY)
    data["semeval_summary"] = read_json_safe(SEMEVAL_RESULTS)
    data["linkedin_summary"]= read_json_safe(LINKEDIN_SUMMARY)
    data["curated_summary"] = read_json_safe(CURATED_SUMMARY)
    return data

DATA = load_all_data()

# ============================================================
# UTILITIES
# ============================================================

def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_number(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def find_metric(obj, keywords):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if any(w in str(key).lower() for w in keywords):
                try:
                    return float(value)
                except Exception:
                    pass
            result = find_metric(value, keywords)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_metric(item, keywords)
            if result is not None:
                return result
    return None


def metric_from_dataframe(df, possible_names):
    if df is None:
        return None
    for col in df.columns:
        col_lower = str(col).lower()
        for name in possible_names:
            if name.lower() in col_lower:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(values):
                    return float(values.iloc[-1])
    return None


def get_logistic_score():
    return metric_from_dataframe(
        DATA["logistic_metrics"], ["accuracy", "score", "f1"]
    )


def get_rf_score():
    return metric_from_dataframe(
        DATA["rf_metrics"], ["accuracy", "mean_test", "score", "f1"]
    )


def get_xgb_score():
    return metric_from_dataframe(
        DATA["xgb_metrics"], ["accuracy", "mean_test", "score", "f1"]
    )


# ============================================================
# SKILL EXTRACTION
# ============================================================

KNOWN_SKILLS = [
    "python", "java", "c++", "c#", "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "artificial intelligence",
    "data science", "data analysis", "data analytics",
    "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "keras",
    "html", "css", "javascript", "typescript", "react", "angular", "vue",
    "node.js", "flask", "django", "fastapi", "rest api",
    "aws", "azure", "google cloud", "cloud computing",
    "docker", "kubernetes", "linux", "git",
    "arduino", "esp32", "iot", "embedded systems", "matlab",
    "verilog", "vlsi", "computer vision", "nlp", "natural language processing",
    "power bi", "tableau", "excel", "r", "scala", "spark", "hadoop",
    "cybersecurity", "networking", "devops", "agile", "project management",
]

# Skills that are substrings of others — check longer first to avoid
# matching "c" inside "c++" or "git" inside "github".
KNOWN_SKILLS_SORTED = sorted(KNOWN_SKILLS, key=len, reverse=True)


def extract_skills(text):
    """Match skills using word-boundary-aware search to avoid substring false positives."""
    text_lower = normalize_text(text).lower()
    found = set()
    for skill in KNOWN_SKILLS_SORTED:
        # Use word boundaries where the skill starts/ends with a word char
        pattern = r'(?<![a-z0-9])' + re.escape(skill) + r'(?![a-z0-9])'
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)


# ============================================================
# LIVE PREDICTION FUNCTIONS
# ============================================================

def lr_predict(skills_text, top_k=5):
    """Logistic Regression prediction from raw text."""
    if LOGISTIC_MODEL_OBJ is None or LOGISTIC_TFIDF_OBJ is None:
        return None
    try:
        X = LOGISTIC_TFIDF_OBJ.transform([skills_text])
        proba = LOGISTIC_MODEL_OBJ.predict_proba(X)[0]
        classes = LOGISTIC_MODEL_OBJ.classes_
        top_idx = np.argsort(proba)[::-1][:top_k]
        return [{"rank": i+1, "career": str(classes[j]), "probability": float(proba[j])}
                for i, j in enumerate(top_idx)]
    except Exception:
        return None


def rf_predict(skills_text, top_k=5):
    """Random Forest prediction from raw text via TF-IDF -> SVD."""
    if RF_MODEL_OBJ is None or RF_TFIDF_OBJ is None or RF_SVD_OBJ is None:
        return None
    try:
        tfidf = RF_TFIDF_OBJ.transform([skills_text])
        svd   = RF_SVD_OBJ.transform(tfidf)
        proba = RF_MODEL_OBJ.predict_proba(svd)[0]
        classes = RF_MODEL_OBJ.classes_
        top_idx = np.argsort(proba)[::-1][:top_k]
        return [{"rank": i+1, "career": str(classes[j]), "probability": float(proba[j])}
                for i, j in enumerate(top_idx)]
    except Exception:
        return None


def xgb_predict_from_text(skills_text, top_k=5):
    """XGBoost prediction from raw text via TF-IDF -> SVD (100 features, no RIASEC)."""
    if XGB_MODEL_OBJ is None or XGB_TFIDF_OBJ is None or XGB_SVD_OBJ is None:
        return None
    if not XGB_CAREER_NAMES:
        return None
    try:
        tfidf = XGB_TFIDF_OBJ.transform([skills_text])
        svd   = XGB_SVD_OBJ.transform(tfidf)
        proba = XGB_MODEL_OBJ.predict_proba(svd)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        return [{"rank": i+1, "career": str(XGB_CAREER_NAMES[j]), "probability": float(proba[j])}
                for i, j in enumerate(top_idx)]
    except Exception:
        return None


def skill_alignment_from_predictions(candidate_skills, career_name):
    """
    Derive career skill profile from RF predictions CSV (skills column).
    Returns (score_pct, matched_list, missing_list).
    """
    candidate = set(s.lower().strip() for s in candidate_skills if s)
    df = DATA.get("rf_predictions")
    if df is None or "true_career" not in df.columns:
        return 0.0, [], []
    career_rows = df[df["true_career"].astype(str).str.lower() == career_name.lower()]
    if career_rows.empty:
        return 0.0, [], []
    # Aggregate skills from the 'skills' column if it exists
    skill_col = "skills" if "skills" in career_rows.columns else None
    if skill_col is None:
        return 0.0, [], []
    all_text = " ".join(career_rows[skill_col].dropna().astype(str).tolist())
    career_skills = set(extract_skills(all_text))
    if not career_skills:
        return 0.0, [], []
    matched = sorted(candidate & career_skills)
    missing = sorted(career_skills - candidate)[:10]
    score = len(matched) / len(career_skills) * 100
    return round(min(score, 100.0), 1), matched, missing


# ============================================================
# RESUME TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):
    try:
        import pypdf
        reader = pypdf.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"[PDF extraction error: {e}]"


def extract_docx_text(uploaded_file):
    try:
        from docx import Document
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def extract_resume_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)
    elif name.endswith(".docx"):
        return extract_docx_text(uploaded_file)
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    return ""


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
# CareerCast
### Milestone 2
**Advanced ML & Recommendation Engine**
""")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Dashboard",
    [
        "Overview",
        "Model Comparison",
        "Resume Prediction",
        "Career Recommendations",
        "SBERT Analytics",
        "Validation",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Existing Milestone 2 artifacts are used. "
    "Models are not retrained when the dashboard starts."
)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<style>
.main-title  { font-size: 34px; font-weight: 700; margin-bottom: 0px; }
.sub-title   { font-size: 15px; color: #666666; margin-top: 0px; }
.metric-card { padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb;
               background: white; box-shadow: 0px 2px 8px rgba(0,0,0,0.04); }
.section-title { font-size: 22px; font-weight: 650; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">CareerCast Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Milestone 2 - Advanced ML and Recommendation Engine</div>', unsafe_allow_html=True)

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown("## Milestone 2 Overview")

    # Read stats from the actual model metric files — not old CSVs
    lr_score  = get_logistic_score()
    rf_score  = get_rf_score()
    xgb_score = get_xgb_score()

    # Training/testing sample counts from RF metrics (most reliable)
    rf_m = DATA["rf_metrics"]
    train_samples = 38400
    test_samples  = 9600
    n_careers     = 96
    if rf_m is not None:
        for col in rf_m.columns:
            cl = col.lower()
            if "train" in cl:
                v = pd.to_numeric(rf_m[col], errors="coerce").dropna()
                if len(v): train_samples = int(v.iloc[-1])
            if "test" in cl and "score" not in cl:
                v = pd.to_numeric(rf_m[col], errors="coerce").dropna()
                if len(v): test_samples = int(v.iloc[-1])
            if "career" in cl:
                v = pd.to_numeric(rf_m[col], errors="coerce").dropna()
                if len(v): n_careers = int(v.iloc[-1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Training Examples", f"{train_samples + test_samples:,}")
    c2.metric("Career Categories",        f"{n_careers:,}")
    c3.metric("Training / Test Split",    f"{train_samples:,} / {test_samples:,}")
    c4.metric("Best Model Accuracy (LR)", f"{lr_score * 100:.3f}%" if lr_score else "N/A")

    st.caption(
        "Numbers above reflect the Milestone 2 model evaluation dataset "
        "(internal held-out test metrics, not real-world generalisation)."
    )

    st.markdown("## Milestone 2 Pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.info("1\n\nCandidate Profile")
    p2.info("2\n\nTF-IDF / SBERT")
    p3.info("3\n\nRF + XGBoost")
    p4.info("4\n\nSkill Alignment")
    p5.success("5\n\nTop-K Careers")

    st.markdown("## Milestone 2 Components")
    components = pd.DataFrame({
        "Component": [
            "Logistic Regression", "Random Forest", "XGBoost",
            "Fine-tuned Sentence-BERT", "Hybrid Recommendation",
            "SemEval Validation", "LinkedIn Transition Validation",
        ],
        "Status": ["Loaded" if LOGISTIC_MODEL_OBJ else "Not found",
                   "Loaded" if RF_MODEL_OBJ else "Not found",
                   "Loaded" if XGB_MODEL_OBJ else "Not found",
                   "Loaded" if SBERT_MODEL_OBJ else "Not found",
                   "Available", "Available", "Available"],
    })
    st.dataframe(components, use_container_width=True, hide_index=True)

# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.markdown("## Model Comparison — Internal Test Accuracy")
    st.caption(
        "Scores are internal held-out test set metrics from the 96-career "
        "Milestone 2 training pipeline. They do not represent real-world accuracy."
    )

    lr_score  = get_logistic_score()
    rf_score  = get_rf_score()
    xgb_score = get_xgb_score()

    chart_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Test Accuracy": [lr_score or 0, rf_score or 0, xgb_score or 0],
    })
    fig = px.bar(chart_df, x="Model", y="Test Accuracy",
                 text=chart_df["Test Accuracy"].apply(lambda x: f"{x*100:.3f}%"),
                 title="Model Test Accuracy Comparison (96 Career Classes)")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Logistic Regression", f"{lr_score*100:.3f}%" if lr_score else "N/A")
        st.caption("TF-IDF → Logistic Regression  |  96 classes")
    with c2:
        st.metric("Random Forest", f"{rf_score*100:.3f}%" if rf_score else "N/A")
        st.caption("TF-IDF → SVD(100) → RF  |  CV-tuned")
    with c3:
        st.metric("XGBoost", f"{xgb_score*100:.3f}%" if xgb_score else "N/A")
        st.caption("TF-IDF → SVD(100) → XGBoost  |  CV-tuned")

    st.markdown("## Random Forest Metrics")
    if DATA["rf_metrics"] is not None:
        st.dataframe(DATA["rf_metrics"], use_container_width=True, hide_index=True)
    else:
        st.info("random_forest_metrics.csv not found.")

    st.markdown("## XGBoost Metrics")
    if DATA["xgb_metrics"] is not None:
        st.dataframe(DATA["xgb_metrics"], use_container_width=True, hide_index=True)
    else:
        st.info("xgboost_metrics.csv not found.")

    st.markdown("## Logistic Regression Metrics")
    if DATA["logistic_metrics"] is not None:
        st.dataframe(DATA["logistic_metrics"], use_container_width=True, hide_index=True)
    else:
        st.info("logistic_metrics.csv not found.")

# ============================================================
# RESUME PREDICTION  (new page)
# ============================================================

elif page == "Resume Prediction":

    st.markdown("## Resume Upload & Career Prediction")
    st.caption(
        "Upload a resume (PDF / DOCX / TXT) or paste text below. "
        "The three trained classifiers will predict the most suitable careers "
        "from the 96-career Milestone 2 model."
    )

    # Status strip
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Logistic Regression", "Ready" if LOGISTIC_MODEL_OBJ else "Not loaded")
    col_s2.metric("Random Forest",        "Ready" if RF_MODEL_OBJ       else "Not loaded")
    col_s3.metric("XGBoost",              "Ready" if XGB_MODEL_OBJ      else "Not loaded")

    if not any([LOGISTIC_MODEL_OBJ, RF_MODEL_OBJ, XGB_MODEL_OBJ]):
        st.error("No classifiers could be loaded. Check that the .pkl files exist in results/.")
        st.stop()

    st.markdown("---")
    top_k = st.slider("Top-K careers to show per model", 3, 10, 5)

    # --- Input ---
    input_tab1, input_tab2 = st.tabs(["Upload File", "Paste Text"])

    resume_text = ""

    with input_tab1:
        uploaded = st.file_uploader(
            "Upload resume (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
        )
        if uploaded is not None:
            with st.spinner("Extracting text..."):
                resume_text = extract_resume_text(uploaded)
            st.success(f"Extracted from: {uploaded.name}")

    with input_tab2:
        pasted = st.text_area(
            "Paste resume text here",
            height=200,
            placeholder="Paste your resume content or skill list here...",
        )
        if pasted.strip():
            resume_text = pasted.strip()

    if not resume_text or resume_text.startswith("["):
        st.info("Upload a file or paste text above, then click Predict.")
        st.stop()

    # --- Preview ---
    with st.expander("Extracted text preview"):
        st.text(resume_text[:1500] + ("..." if len(resume_text) > 1500 else ""))

    # --- Skills ---
    skills = extract_skills(resume_text)
    st.markdown(f"### Detected Skills ({len(skills)} found)")
    if skills:
        st.write("  ".join(f"`{s}`" for s in skills))
    else:
        st.warning(
            "No skills matched the vocabulary. "
            "The full resume text will still be used as model input."
        )

    # Use skills joined as the model input text for consistency
    # with training (training data was 'skill | skill | ...' format)
    model_input = " | ".join(skills) if skills else resume_text[:2000]

    # --- Predict ---
    if st.button("Predict Careers", type="primary", use_container_width=True):

        with st.spinner("Running classifiers..."):
            lr_results  = lr_predict(model_input, top_k)
            rf_results  = rf_predict(model_input, top_k)
            xgb_results = xgb_predict_from_text(model_input, top_k)

        st.session_state["resume_lr"]  = lr_results
        st.session_state["resume_rf"]  = rf_results
        st.session_state["resume_xgb"] = xgb_results
        st.session_state["resume_skills"] = skills

    lr_results  = st.session_state.get("resume_lr")
    rf_results  = st.session_state.get("resume_rf")
    xgb_results = st.session_state.get("resume_xgb")
    cached_skills = st.session_state.get("resume_skills", skills)

    if lr_results is None and rf_results is None and xgb_results is None:
        st.info("Click 'Predict Careers' to run the models.")
        st.stop()

    # --- Results ---
    st.markdown("---")
    st.markdown("## Prediction Results")
    st.caption(
        "Scores are model probability outputs (0–1). "
        "Higher = more likely according to the classifier. "
        "These are not calibrated confidence values."
    )

    col_lr, col_rf, col_xgb = st.columns(3)

    def render_predictions(col, title, results):
        with col:
            st.markdown(f"**{title}**")
            if results is None:
                st.warning("Model not available.")
                return
            for r in results:
                st.write(f"{r['rank']}. {r['career']}")
                st.progress(min(r['probability'], 1.0))
                st.caption(f"Score: {r['probability']:.4f}")

    render_predictions(col_lr,  "Logistic Regression", lr_results)
    render_predictions(col_rf,  "Random Forest",        rf_results)
    render_predictions(col_xgb, "XGBoost",              xgb_results)

    # --- Top combined recommendation ---
    st.markdown("---")
    st.markdown("### Combined Top Recommendation")

    all_results = {}
    for results in [lr_results, rf_results, xgb_results]:
        if results:
            for r in results:
                career = r["career"]
                all_results[career] = all_results.get(career, 0.0) + r["probability"]

    if all_results:
        top_career = max(all_results, key=all_results.get)
        st.success(f"**Most recommended career: {top_career}**")
        st.caption(
            "Combined by summing probability scores across all available models. "
            "Equal weight per model."
        )

        # Skill alignment for top career
        score, matched, missing = skill_alignment_from_predictions(cached_skills, top_career)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Skill Alignment", f"{score:.1f}%",
                      help="Matched skills / career profile skills derived from training data. "
                           "Not the same as model probability.")
            if matched:
                st.caption("Matched: " + ", ".join(matched))
        with col_b:
            if missing:
                st.markdown("**Suggested skill gaps:**")
                for s in missing[:8]:
                    st.write(f"• {s}")

        # Bar chart of top careers by combined score
        top_n = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        chart_df = pd.DataFrame(top_n, columns=["Career", "Combined Score"])
        fig = px.bar(chart_df, x="Combined Score", y="Career",
                     orientation="h", title="Combined Career Ranking")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CAREER RECOMMENDATIONS  (existing page — unchanged logic)
# ============================================================

elif page == "Career Recommendations":

    st.markdown("## Top-K Career Recommendation Engine")
    st.caption("Enter your profile manually to generate recommendations via the hybrid engine.")

    col1, col2 = st.columns(2)
    with col1:
        skills_input = st.text_input(
            "Skills",
            value="Python, SQL, Machine Learning, Pandas, NumPy, Data Analysis, TensorFlow",
        )
        education = st.text_input("Education", value="B.E. Electronics and Communication Engineering")
        experience = st.text_input("Experience", value="6 months internship in Python, data analysis and ML")

    with col2:
        job_description = st.text_area(
            "Career / Job Description",
            value="Develop ML models, analyze datasets, build predictive systems.",
            height=120,
        )
        top_k = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Generate Career Recommendations", type="primary", use_container_width=True):
        with st.spinner("Running classifiers..."):
            input_text = skills_input + " " + education + " " + experience + " " + job_description
            lr  = lr_predict(input_text, top_k)
            rf  = rf_predict(input_text, top_k)
            xgb = xgb_predict_from_text(input_text, top_k)

        if lr is None and rf is None and xgb is None:
            st.error("No classifier results available. Check that model files are loaded.")
        else:
            st.markdown("## Top Career Recommendations")
            c1, c2, c3 = st.columns(3)
            def render_col(col, title, results):
                with col:
                    st.markdown(f"**{title}**")
                    if results:
                        for r in results:
                            with st.container():
                                st.write(f"{r['rank']}. {r['career']}")
                                st.caption(f"Score: {r['probability']:.4f}")
                    else:
                        st.warning("Unavailable")
            render_col(c1, "Logistic Regression", lr)
            render_col(c2, "Random Forest", rf)
            render_col(c3, "XGBoost", xgb)

# ============================================================
# SBERT ANALYTICS
# ============================================================

elif page == "SBERT Analytics":

    st.markdown("## Fine-tuned Sentence-BERT Analytics")

    c1, c2, c3 = st.columns(3)
    c1.metric("SBERT Model", "Loaded" if SBERT_MODEL_OBJ else "Not found")
    if CAREER_EMBEDDINGS is not None:
        c2.metric("Career Embeddings", f"{len(CAREER_EMBEDDINGS):,}")
        c3.metric("Embedding Dimension",
                  str(CAREER_EMBEDDINGS.shape[1]) if CAREER_EMBEDDINGS.ndim > 1 else "N/A")
    else:
        c2.metric("Career Embeddings", "Unavailable")
        c3.metric("Embedding Dimension", "N/A")

    st.markdown("## Pre-trained vs Fine-tuned SBERT")
    comparison = DATA["sbert_comparison"]
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        score_cols = [c for c in ["original_score", "finetuned_score"] if c in comparison.columns]
        if len(score_cols) == 2:
            plot_df = comparison[score_cols].reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df.index+1, y=plot_df["original_score"],
                                     mode="lines+markers", name="Original SBERT"))
            fig.add_trace(go.Scatter(x=plot_df.index+1, y=plot_df["finetuned_score"],
                                     mode="lines+markers", name="Fine-tuned SBERT"))
            fig.update_layout(title="SBERT Similarity Comparison",
                              xaxis_title="Rank", yaxis_title="Similarity Score")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SBERT comparison CSV not found.")

    st.markdown("## t-SNE Visualization: Career Embeddings")
    if CAREER_EMBEDDINGS is not None:
        try:
            from sklearn.manifold import TSNE
            embeddings = np.asarray(CAREER_EMBEDDINGS)
            labels_df = read_csv_safe(SBERT_LABELS)
            labels = labels_df["Career"].astype(str).values if (
                labels_df is not None and "Career" in labels_df.columns
            ) else np.arange(len(embeddings)).astype(str)

            @st.cache_data
            def calculate_tsne(arr):
                perp = min(30, max(5, len(arr) // 20))
                return TSNE(n_components=2, random_state=42, perplexity=perp,
                            init="pca", learning_rate="auto").fit_transform(arr)

            coords = calculate_tsne(embeddings)
            tsne_df = pd.DataFrame({"TSNE_1": coords[:, 0], "TSNE_2": coords[:, 1], "Career": labels})
            fig = px.scatter(tsne_df, x="TSNE_1", y="TSNE_2", color="Career",
                             hover_name="Career", title="Fine-tuned SBERT Career Embedding Space")
            fig.update_traces(marker={"size": 6, "opacity": 0.7})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each point is a career embedding. Nearby points are semantically similar.")
        except Exception:
            st.warning("t-SNE visualization could not be calculated.")
    else:
        st.warning("Fine-tuned career embeddings were not found.")

# ============================================================
# VALIDATION
# ============================================================

elif page == "Validation":

    st.markdown("## Milestone 2 Validation")

    st.markdown("## SemEval Semantic Similarity Validation")
    st.caption(
        "SemEval-2017 STS track5 (en-en, 250 pairs) validates general "
        "sentence-similarity quality of the SBERT model. "
        "It is not a career-classification benchmark."
    )
    semeval = DATA["semeval_summary"]
    if semeval:
        sem_pearson  = find_metric(semeval, ["pearson"])
        sem_spearman = find_metric(semeval, ["spearman"])
        sc1, sc2 = st.columns(2)
        if sem_pearson  is not None: sc1.metric("Pearson r",   f"{sem_pearson:.4f}")
        if sem_spearman is not None: sc2.metric("Spearman rho", f"{sem_spearman:.4f}")
        with st.expander("View SemEval details"):
            st.json(semeval)
    else:
        st.info("SemEval results JSON not found.")

    st.markdown("## LinkedIn Transition Validation")
    linkedin = DATA["linkedin_summary"]
    if linkedin:
        top1 = find_metric(linkedin, ["top1", "top_1"])
        top3 = find_metric(linkedin, ["top3", "top_3"])
        top5 = find_metric(linkedin, ["top5", "top_5"])
        l1, l2, l3 = st.columns(3)
        if top1 is not None:
            l1.metric("LinkedIn Top-1", f"{top1*100:.2f}%" if top1 <= 1 else f"{top1:.2f}%")
        if top3 is not None:
            l2.metric("LinkedIn Top-3", f"{top3*100:.2f}%" if top3 <= 1 else f"{top3:.2f}%")
        if top5 is not None:
            l3.metric("LinkedIn Top-5", f"{top5*100:.2f}%" if top5 <= 1 else f"{top5:.2f}%")
        with st.expander("View LinkedIn details"):
            st.json(linkedin)
    else:
        st.info("LinkedIn validation summary not found.")

    st.markdown("## Curated Candidate Results")
    curated = DATA["curated_results"]
    if curated is not None:
        st.dataframe(curated, use_container_width=True, hide_index=True)
    else:
        st.info("Curated results CSV not found.")

    st.markdown("## Hybrid Recommender Validation")
    hybrid = DATA["hybrid_summary"]
    if hybrid:
        with st.expander("View hybrid evaluation summary", expanded=True):
            st.json(hybrid)

    alpha = DATA["alpha_results"]
    if alpha is not None:
        st.markdown("### Hybrid Weight Tuning")
        st.dataframe(alpha, use_container_width=True, hide_index=True)
        if "alpha" in alpha.columns and "top1_accuracy" in alpha.columns:
            fig = px.line(alpha, x="alpha",
                          y=[c for c in ["top1_accuracy", "top3_accuracy", "top5_accuracy"]
                             if c in alpha.columns],
                          markers=True, title="Hybrid Accuracy vs Alpha")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("CareerCast | Milestone 2 | ML artifacts loaded from results/ directory")
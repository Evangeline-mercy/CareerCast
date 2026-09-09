"""CareerCast professional Streamlit career-intelligence dashboard.

Run the FastAPI service first, then start this app from the project root:
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from streamlit_app.report_builder import build_career_report
from streamlit_app.analytics import build_cohort_result_row, prepare_cohort, summarize_cohort
from streamlit_app.milestone2_dashboard import render_milestone2_dashboard
from streamlit_app.auth import access_token, clear_session, configure_auth_environment, render_auth_gate


API_BASE_URL = os.getenv("CAREERCAST_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 120

st.set_page_config(
    page_title="CareerCast | Career Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_auth_environment()


@st.cache_resource(show_spinner="Preparing CareerCast models for the first launch...")
def prepare_cloud_runtime() -> bool:
    """Start the API inside Streamlit only when cloud deployment requests it."""
    if os.getenv("CAREERCAST_EMBEDDED_API", "0").lower() not in {"1", "true", "yes"}:
        return False
    from deployment.bootstrap import start_embedded_api

    start_embedded_api()
    return True


prepare_cloud_runtime()

st.markdown(
    """
    <style>
      :root {--cc-blue:#0b63e5; --cc-green:#0f9f6e; --cc-red:#e5484d; --cc-ink:#111827;}
      .stApp {background:linear-gradient(135deg,#f3f8ff 0%,#f8fbff 52%,#eefbf6 100%); color:var(--cc-ink);}
      [data-testid="stSidebar"] {background:linear-gradient(180deg,#edf5ff 0%,#ffffff 55%,#eefbf6 100%); border-right:1px solid #d7e5f7;}
      [data-testid="stSidebar"] * {color:#172033;}
      [data-testid="stSidebar"] [data-testid="stButton"] button {border-radius:10px;}
      [data-testid="stSidebar"] div[role="radiogroup"] label {padding:.5rem .6rem; border-radius:10px; margin:.08rem 0;}
      [data-testid="stSidebar"] div[role="radiogroup"] label:hover {background:#dcecff;}
      [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background:#d8eaff; color:#075ac6; font-weight:700;}
      .block-container {padding-top:2rem; max-width:1450px;}
      .hero {padding:1.35rem 1.5rem; border:1px solid #b9d8ff; border-radius:18px;
             color:#0f2747; background:linear-gradient(115deg,#dcebff 0%,#f2f8ff 50%,#dcf8eb 100%); margin-bottom:1.15rem;
             box-shadow:0 10px 26px rgba(11,99,229,.10);}
      .hero h1 {margin:0; font-size:2rem; letter-spacing:-.02em;}
      .hero h1 span {color:var(--cc-blue);}
      .hero p {margin:.35rem 0 0; color:#667085;}
      .cc-card {background:linear-gradient(145deg,#ffffff,#f7faff); border:1px solid #cfdef0; border-radius:16px;
                padding:1.15rem 1.2rem; min-height:132px; box-shadow:0 5px 16px rgba(15,23,42,.045);}
      .cc-card:hover {transform:translateY(-2px); box-shadow:0 10px 24px rgba(11,99,229,.10); transition:.18s ease;}
      .cc-blue {background:linear-gradient(145deg,#e7f2ff,#ffffff); border-top:4px solid #0b63e5;}
      .cc-red {background:linear-gradient(145deg,#fff0f1,#ffffff); border-top:4px solid #e5484d;}
      .cc-green {background:linear-gradient(145deg,#e9faf3,#ffffff); border-top:4px solid #0f9f6e;}
      .cc-dark {background:linear-gradient(145deg,#edf1f6,#ffffff); border-top:4px solid #25334a;}
      .cc-card h3 {margin:.35rem 0; font-size:1.12rem;}
      .cc-card p {color:#667085; margin:.25rem 0;}
      .cc-status {font-size:1.55rem; font-weight:750; color:#111827;}
      .cc-icon {font-size:1.45rem;}
      div[data-testid="stMetric"] {background:linear-gradient(145deg,#ffffff,#edf5ff); border:1px solid #cfe0f4; border-radius:14px;
                                   padding:1rem; box-shadow:0 4px 14px rgba(15,23,42,.04);}
      div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] {
          background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:.35rem;}
      .stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
          background:var(--cc-blue); border-color:var(--cc-blue); border-radius:10px;}
      .chip {display:inline-block; padding:.3rem .65rem; margin:.15rem; border-radius:999px;
             background:#e8f2ff; color:#074ea8; font-size:.85rem; font-weight:600;}
      .priority-high {color:#b91c1c; font-weight:700;}
      .priority-medium {color:#b45309; font-weight:700;}
      .priority-low {color:#047857; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {access_token()}"} if access_token() else {}
    response = requests.get(f"{API_BASE_URL}{path}", headers=headers, timeout=15)
    if response.status_code == 401:
        clear_session()
        raise RuntimeError("Your session expired. Please sign in again.")
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {access_token()}"} if access_token() else {}
    response = requests.post(
        f"{API_BASE_URL}{path}", json=payload, headers=headers, timeout=REQUEST_TIMEOUT
    )
    if response.status_code == 401:
        clear_session()
        raise RuntimeError("Your session expired. Please sign in again.")
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise RuntimeError(f"API {response.status_code}: {detail}")


def extract_uploaded_text(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(uploaded_file).pages)
    if suffix == ".docx":
        from docx import Document

        return "\n".join(paragraph.text for paragraph in Document(uploaded_file).paragraphs)
    raise ValueError("Only PDF, DOCX and TXT files are supported.")


def probability_chart(recommendations: list[dict[str, Any]]):
    frame = pd.DataFrame(recommendations)
    frame["Probability (%)"] = frame["ensemble_score"] * 100
    chart = px.bar(
        frame.sort_values("Probability (%)"),
        x="Probability (%)",
        y="career",
        orientation="h",
        color="Probability (%)",
        color_continuous_scale=["#8fc2ff", "#0b63e5"],
        labels={"career": "Career"},
        text=frame.sort_values("Probability (%)")["Probability (%)"].map(lambda value: f"{value:.2f}%"),
    )
    chart.update_layout(coloraxis_showscale=False, height=max(330, len(frame) * 52), margin=dict(l=10, r=20, t=20, b=10))
    chart.update_traces(textposition="outside")
    return chart


def render_cohort_workspace(show_results: bool = False, show_download: bool = False) -> None:
    st.markdown("## 📂 Cohort Analytics")
    st.caption(
        "Upload a CSV containing a `skills` column and an optional `name` column. "
        "CareerCast analyses up to 50 profiles per run."
    )
    template = pd.DataFrame(
        [
            {"name": "Candidate 1", "skills": "Python, SQL, pandas"},
            {"name": "Candidate 2", "skills": "HTML, CSS, JavaScript"},
        ]
    )
    st.download_button(
        "Download cohort CSV template",
        template.to_csv(index=False).encode("utf-8"),
        "CareerCast_Cohort_Template.csv",
        "text/csv",
    )
    cohort_file = st.file_uploader("Upload cohort CSV", type=["csv"], key="cohort_csv")
    if cohort_file is not None:
        try:
            cohort = prepare_cohort(pd.read_csv(cohort_file))
            st.dataframe(cohort, width="stretch", hide_index=True)
        except ValueError as exc:
            st.error(str(exc))
            return

        if st.button("Analyse Cohort", type="primary", width="stretch"):
            rows = []
            generated_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
            progress = st.progress(0, text="Analysing cohort profiles...")
            try:
                for position, record in enumerate(cohort.to_dict("records"), start=1):
                    recommendation = api_post(
                        "/recommend", {"skills_text": record["skills"], "top_k": 3}
                    )
                    ranked = recommendation["recommendations"]
                    top = ranked[0]
                    gap = api_post(
                        "/gap-report",
                        {
                            "skills_text": record["skills"],
                            "target_career": top["career"],
                            "top_k_careers": 1,
                        },
                    )["gap_analysis"][0]
                    rows.append(build_cohort_result_row(record, ranked, gap, generated_at))
                    progress.progress(position / len(cohort), text=f"Analysed {position} of {len(cohort)}")
                st.session_state["cohort_result"] = pd.DataFrame(rows)
                progress.empty()
            except (requests.RequestException, RuntimeError, KeyError, IndexError) as exc:
                progress.empty()
                st.error(f"Cohort analysis failed: {exc}")

    result = st.session_state.get("cohort_result")
    if show_results and result is not None and not result.empty:
        summary = summarize_cohort(result)
        m1, m2, m3 = st.columns(3)
        m1.metric("Profiles analysed", summary["profile_count"])
        m2.metric("Career paths identified", summary["career_count"])
        m3.metric("Mean skill alignment", f"{summary['mean_alignment']:.2f}%")
        career_counts = summary["career_distribution"]
        chart = px.bar(
            career_counts,
            x="Profiles",
            y="Predicted career",
            orientation="h",
            color="Profiles",
            color_continuous_scale=["#9bd9c4", "#0b63e5"],
            title="Cohort career distribution",
        )
        chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(chart, width="stretch")
        st.dataframe(result, width="stretch", hide_index=True)
        if show_download:
            st.download_button(
                "Download cohort results",
                result.to_csv(index=False).encode("utf-8"),
                "CareerCast_Cohort_Results.csv",
                "text/csv",
                type="primary",
                width="stretch",
            )


def render_overview() -> None:
    email = st.session_state.get("auth_user_email", "CareerCast user")
    name = str(email).split("@", 1)[0].replace(".", " ").replace("_", " ").title()
    st.markdown(
        f'<div class="hero"><h1>Welcome to <span>CareerCast</span></h1>'
        f'<p>Hello, {name}. Turn your skills into clear career possibilities.</p>'
        f'<p><strong>“Your skills are the beginning; your possibilities are limitless.”</strong></p></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    cards = [
        ("👩‍💻", "Career Profiles", "Available", "Explore roles that match your profile."),
        ("🎯", "Skill-Gap Intelligence", "Ready", "See strengths and missing career skills."),
        ("📊", "Active Models", "3", "Logistic Regression, Random Forest and XGBoost."),
        ("🛡️", "Security", "Protected", "Supabase authentication is active."),
    ]
    card_classes = ["cc-blue", "cc-red", "cc-green", "cc-dark"]
    for column, (icon, title, status, detail), card_class in zip(columns, cards, card_classes):
        with column:
            st.markdown(
                f'<div class="cc-card {card_class}"><div class="cc-icon">{icon}</div><h3>{title}</h3>'
                f'<div class="cc-status">{status}</div><p>{detail}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("### Explore your workspace")
    columns = st.columns(4)
    workspaces = [
        ("🔍 Career Analysis", "Upload a resume or enter skills to receive ranked career matches and skill-gap guidance."),
        ("💡 Career Recommendations", "Review ranked career possibilities produced by the model ensemble."),
        ("📊 Skill Gap Analysis", "Identify matched skills, missing skills, priorities and suggested actions."),
        ("⚖️ Career Comparison", "Compare two recommended careers using alignment and gap evidence."),
    ]
    for column, (title, detail) in zip(columns, workspaces):
        with column:
            st.markdown(f'<div class="cc-card"><h3>{title}</h3><p>{detail}</p></div>', unsafe_allow_html=True)
    st.markdown("### How CareerCast works")
    steps = st.columns(3)
    for column, (number, title, detail) in zip(steps, [
        ("1", "Add profile", "Provide resume text, skills, experience or career goals."),
        ("2", "Run intelligence", "The models rank career paths and compare relevant skill requirements."),
        ("3", "Explore results", "Review recommendations, gaps, comparisons and a downloadable report."),
    ]):
        with column:
            st.markdown(f'<div class="cc-card"><span class="cc-status">{number}</span><h3>{title}</h3><p>{detail}</p></div>', unsafe_allow_html=True)


def render_account() -> None:
    st.markdown("## ⚙️ Account")
    st.markdown('<div class="cc-card"><h3>Signed-in account</h3><p>Your CareerCast workspace is protected by Supabase Authentication.</p></div>', unsafe_allow_html=True)
    st.text_input("Email", value=st.session_state.get("auth_user_email", ""), disabled=True)
    st.success("Email authentication and protected API access are active.")
    st.markdown(
        '<div class="hero"><h1>Thank you for using <span>CareerCast</span></h1>'
        '<p>Keep learning, keep growing, and keep moving towards your future.</p></div>',
        unsafe_allow_html=True,
    )


def career_result() -> dict[str, Any] | None:
    result = st.session_state.get("career_result")
    if not result:
        st.info("Run Career Analysis first. Its results will automatically appear here.")
        return None
    return result


def render_career_analysis() -> None:
    st.markdown('<div class="hero"><h1>🔍 Career <span>Analysis</span></h1><p>Upload a resume or enter your skills to discover matching career paths.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a resume", type=["pdf", "docx", "txt"], help="PDF, DOCX or TXT")
    uploaded_text = ""
    if uploaded is not None:
        try:
            uploaded_text = extract_uploaded_text(uploaded).strip()
            st.success(f"Extracted text from {uploaded.name}")
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
    previous = st.session_state.get("career_result", {}).get("profile_text", "")
    profile_text = st.text_area(
        "Profile or skills", value=uploaded_text or previous, height=190,
        placeholder="Example: Python, SQL, pandas, machine learning, statistics...",
        help="Review the extracted text or enter a skill profile manually.",
    )
    left, right = st.columns(2)
    with left:
        top_k = st.slider("Number of career recommendations", 3, 10, 5)
    with right:
        target = st.text_input("Target career (optional)", placeholder="Example: Data Scientist")
    if st.button("Analyse Career Profile", type="primary", width="stretch"):
        if not profile_text.strip():
            st.warning("Enter profile text or upload a resume first.")
            return
        try:
            with st.spinner("Running CareerCast models and skill-gap analysis..."):
                prediction = api_post("/predict", {"skills_text": profile_text, "top_k": top_k})
                recommendation = api_post("/recommend", {"skills_text": profile_text, "top_k": top_k})
                gap_payload: dict[str, Any] = {"skills_text": profile_text, "top_k_careers": 1}
                if target.strip():
                    gap_payload["target_career"] = target.strip()
                else:
                    ranked = recommendation.get("recommendations", [])
                    if ranked:
                        gap_payload["target_career"] = ranked[0]["career"]
                gap_report = api_post("/gap-report", gap_payload)
                model_info = api_get("/models/info")
            st.session_state["career_result"] = {
                "profile_text": profile_text, "prediction": prediction,
                "recommendation": recommendation, "gap_report": gap_report,
                "model_info": model_info,
            }
            st.success("Analysis completed. Open the result dashboards from the sidebar.")
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Analysis failed: {exc}")
    result = st.session_state.get("career_result")
    if result:
        top = result["recommendation"].get("recommendations", [])
        primary = top[0] if top else {}
        st.markdown("### Latest analysis")
        left, right = st.columns(2)
        left.metric("Predicted career", primary.get("career", "Unavailable"))
        right.metric("Ensemble score", f"{float(primary.get('ensemble_score', 0)) * 100:.2f}%")


def render_recommendations() -> None:
    st.markdown("## 💡 Career Recommendations")
    st.caption("Ranked career possibilities produced by the three-model ensemble.")
    result = career_result()
    if not result:
        return
    recommendations = result["recommendation"].get("recommendations", [])
    if not recommendations:
        st.warning("No recommendations are available for the latest analysis.")
        return
    st.plotly_chart(probability_chart(recommendations), width="stretch")
    display = pd.DataFrame(recommendations).rename(columns={
        "rank": "Rank", "career": "Career", "ensemble_score": "Ensemble score",
        "lr_probability": "LR", "rf_probability": "RF", "xgb_probability": "XGBoost",
    })
    st.dataframe(display, width="stretch", hide_index=True)


def render_skill_gap() -> None:
    st.markdown("## 📊 Skill Gap Analysis")
    st.caption("Understand existing strengths, missing skills, priority and suggested action.")
    result = career_result()
    if not result:
        return
    gap_report = result["gap_report"]
    gaps = gap_report.get("gap_analysis", [])
    primary = gaps[0] if gaps else {}
    st.markdown(f"### Target: {gap_report.get('target_career', 'Unavailable')}")
    st.markdown("**Matched skills**")
    matched = primary.get("matched_skills", [])
    if matched:
        st.markdown("".join(f'<span class="chip">{skill}</span>' for skill in matched), unsafe_allow_html=True)
    else:
        st.info("No matched skills were detected for this target.")
    missing = primary.get("missing_skills", [])
    st.markdown("### Missing skills and actions")
    if missing:
        frame = pd.DataFrame(missing).rename(columns={"skill": "Skill", "weight": "Weight", "priority": "Priority", "suggestion": "Action"})
        st.dataframe(frame, width="stretch", hide_index=True)
        summary = primary.get("priority_summary", {})
        columns = st.columns(3)
        columns[0].metric("High priority", summary.get("High", 0))
        columns[1].metric("Medium priority", summary.get("Medium", 0))
        columns[2].metric("Low priority", summary.get("Low", 0))
    else:
        st.success("No missing skills were identified.")


def render_career_comparison() -> None:
    st.markdown("## ⚖️ Career Comparison")
    st.caption("Compare the skill alignment and gaps of two recommended careers.")
    result = career_result()
    if not result:
        return
    recommendations = result["recommendation"].get("recommendations", [])
    options = [item["career"] for item in recommendations]
    if len(options) < 2:
        st.info("At least two recommendations are required for comparison.")
        return
    left, right = st.columns(2)
    with left:
        career_a = st.selectbox("Career A", options, index=0)
    with right:
        career_b = st.selectbox("Career B", options, index=1)
    if career_a == career_b:
        st.warning("Select two different careers.")
    elif st.button("Compare Careers", type="primary", width="stretch"):
        try:
            with st.spinner("Comparing career skill requirements..."):
                comparison = [api_post("/gap-report", {
                    "skills_text": result["profile_text"], "target_career": career,
                    "top_k_careers": 1,
                })["gap_analysis"][0] for career in (career_a, career_b)]
            st.session_state["career_comparison"] = comparison
        except (requests.RequestException, RuntimeError, KeyError, IndexError) as exc:
            st.error(f"Career comparison failed: {exc}")
    comparison = st.session_state.get("career_comparison")
    if comparison:
        frame = pd.DataFrame([{
            "Career": item["career"], "Skill alignment (%)": item["alignment_score"],
            "Matched skills": len(item["matched_skills"]), "Missing skills": len(item["missing_skills"]),
            "High-priority gaps": item["priority_summary"].get("High", 0),
        } for item in comparison])
        st.plotly_chart(px.bar(frame, x="Career", y="Skill alignment (%)", color="Career", range_y=[0, 100], title="Skill alignment comparison"), width="stretch")
        st.dataframe(frame, width="stretch", hide_index=True)


def render_individual_report() -> None:
    st.markdown("## 📄 Individual Report")
    st.caption("Generate a PDF from the latest individual career analysis.")
    result = career_result()
    if not result:
        return
    try:
        pdf = build_career_report(result["profile_text"], result["prediction"], result["recommendation"], result["gap_report"], result["model_info"])
        st.download_button("Download CareerCast PDF Report", pdf, "CareerCast_Career_Report.pdf", "application/pdf", type="primary", width="stretch")
        st.info("This report includes predictions, ranked careers, weighted skill gaps, actions and model information.")
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")


def render_cohort_report() -> None:
    st.markdown("## 📥 Cohort Report Download")
    st.caption("Download results from the latest Cohort Analytics run.")
    result = st.session_state.get("cohort_result")
    if result is None or result.empty:
        st.info("Run Cohort Analytics first. Its results will automatically appear here.")
        return
    st.success("Your latest cohort analysis is ready to download.")
    st.caption(f"The report contains {len(result)} analysed profiles, input skills, top-three recommendations, confidence, alignment and priority gaps.")
    st.download_button("Download Detailed Cohort Report (CSV)", result.to_csv(index=False).encode("utf-8-sig"), "CareerCast_Detailed_Cohort_Report.csv", "text/csv", type="primary", width="stretch")


st.sidebar.markdown("# 📊 CareerCast")
st.sidebar.caption("Your skills. A brighter tomorrow.")
st.sidebar.markdown("---")

if not render_auth_gate():
    st.stop()

try:
    api_get("/health")
    st.sidebar.success("API connected")
except requests.RequestException:
    st.sidebar.error("API is not running")
    st.error("Start the CareerCast FastAPI service on port 8000, then refresh this page.")
    st.code("python -m uvicorn api.main:app --port 8000")
    st.stop()

pages = [
    "🏠 Overview", "🔍 Career Analysis", "💡 Career Recommendations",
    "📊 Skill Gap Analysis", "⚖️ Career Comparison", "🧠 Model Insights",
    "📂 Cohort Analytics", "📄 Individual Report", "📥 Cohort Report Download", "⚙️ Account",
]
workspace = st.sidebar.radio("Navigation", pages)

if workspace == "🏠 Overview":
    render_overview()
elif workspace == "🔍 Career Analysis":
    render_career_analysis()
elif workspace == "💡 Career Recommendations":
    render_recommendations()
elif workspace == "📊 Skill Gap Analysis":
    render_skill_gap()
elif workspace == "⚖️ Career Comparison":
    render_career_comparison()
elif workspace == "🧠 Model Insights":
    try:
        render_milestone2_dashboard(api_get, api_post)
    except (requests.RequestException, RuntimeError, AttributeError, TypeError) as exc:
        st.error(f"Model Insights could not be loaded: {exc}")
elif workspace == "📂 Cohort Analytics":
    render_cohort_workspace(show_results=True, show_download=False)
elif workspace == "📄 Individual Report":
    render_individual_report()
elif workspace == "📥 Cohort Report Download":
    render_cohort_report()
elif workspace == "⚙️ Account":
    render_account()

st.markdown("---")
st.caption("CareerCast | Secure career-intelligence workspace")

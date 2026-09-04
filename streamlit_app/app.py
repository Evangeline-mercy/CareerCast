"""CareerCast Milestone 4 Streamlit review and cohort analytics UI.

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
from streamlit_app.analytics import prepare_cohort, summarize_cohort


API_BASE_URL = os.getenv("CAREERCAST_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 120

st.set_page_config(
    page_title="CareerCast | Career Intelligence",
    page_icon="CC",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
      .stApp {background: linear-gradient(135deg, #eef2ff 0%, #ecfeff 45%, #fff7ed 100%);}
      [data-testid="stSidebar"] {background: linear-gradient(180deg, #1e1b4b, #0f766e);}
      [data-testid="stSidebar"] * {color: #ffffff;}
      .hero {padding: 1.35rem 1.6rem; border-radius: 20px; color: white;
             background: linear-gradient(120deg, #4338ca, #0f766e); margin-bottom: 1rem;
             box-shadow: 0 12px 30px rgba(49,46,129,.18);}
      .hero h1 {margin: 0; font-size: 2.25rem;}
      .hero p {margin: .35rem 0 0; opacity: .92;}
      .chip {display:inline-block; padding:.3rem .65rem; margin:.15rem; border-radius:999px;
             background:#e0e7ff; color:#312e81; font-size:.85rem; font-weight:600;}
      .priority-high {color:#b91c1c; font-weight:700;}
      .priority-medium {color:#b45309; font-weight:700;}
      .priority-low {color:#047857; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
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
        color_continuous_scale=["#0f766e", "#4338ca"],
        labels={"career": "Career"},
        text=frame.sort_values("Probability (%)")["Probability (%)"].map(lambda value: f"{value:.2f}%"),
    )
    chart.update_layout(coloraxis_showscale=False, height=max(330, len(frame) * 52), margin=dict(l=10, r=20, t=20, b=10))
    chart.update_traces(textposition="outside")
    return chart


def render_cohort_workspace() -> None:
    st.markdown("## Cohort Analytics")
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
            st.dataframe(cohort, use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.error(str(exc))
            return

        if st.button("Analyse Cohort", type="primary", use_container_width=True):
            rows = []
            progress = st.progress(0, text="Analysing cohort profiles...")
            try:
                for position, record in enumerate(cohort.to_dict("records"), start=1):
                    recommendation = api_post(
                        "/recommend", {"skills_text": record["skills"], "top_k": 1}
                    )
                    top = recommendation["recommendations"][0]
                    gap = api_post(
                        "/gap-report",
                        {
                            "skills_text": record["skills"],
                            "target_career": top["career"],
                            "top_k_careers": 1,
                        },
                    )["gap_analysis"][0]
                    rows.append(
                        {
                            "Name": record["name"],
                            "Predicted career": top["career"],
                            "Career score (%)": round(top["ensemble_score"] * 100, 2),
                            "Skill alignment (%)": round(gap["alignment_score"], 2),
                            "High-priority gaps": gap["priority_summary"].get("High", 0),
                        }
                    )
                    progress.progress(position / len(cohort), text=f"Analysed {position} of {len(cohort)}")
                st.session_state["cohort_result"] = pd.DataFrame(rows)
                progress.empty()
            except (requests.RequestException, RuntimeError, KeyError, IndexError) as exc:
                progress.empty()
                st.error(f"Cohort analysis failed: {exc}")

    result = st.session_state.get("cohort_result")
    if result is not None and not result.empty:
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
            color_continuous_scale=["#0f766e", "#4338ca"],
            title="Cohort career distribution",
        )
        chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download cohort results",
            result.to_csv(index=False).encode("utf-8"),
            "CareerCast_Cohort_Results.csv",
            "text/csv",
            type="primary",
            use_container_width=True,
        )


st.sidebar.markdown("# CareerCast")
st.sidebar.markdown("**Milestone 4 Intelligence Suite**")
st.sidebar.caption("Individual • Cohort • Comparison • PDF")
st.sidebar.markdown("---")
st.sidebar.code(API_BASE_URL, language=None)

try:
    health = api_get("/health")
    st.sidebar.success("API connected")
    st.sidebar.caption(f"Career profiles: {health.get('career_profiles_loaded', 0)}")
    st.sidebar.caption(f"Gap profiles: {health.get('gap_profiles_loaded', 0)}")
except requests.RequestException:
    st.sidebar.error("API is not running")
    st.error("Start the CareerCast FastAPI service on port 8000, then refresh this page.")
    st.code("python -m uvicorn api.main:app --port 8000")
    st.stop()

st.markdown(
    '<div class="hero"><h1>CareerCast</h1><p>AI-powered career prediction and weighted skill-gap intelligence</p></div>',
    unsafe_allow_html=True,
)

workspace = st.sidebar.radio("Workspace", ["Individual Review", "Cohort Analytics"])
if workspace == "Cohort Analytics":
    render_cohort_workspace()
    st.markdown("---")
    st.caption("CareerCast | Milestone 4 | Cohort decision-support prototype")
    st.stop()

uploaded = st.file_uploader("Upload a resume", type=["pdf", "docx", "txt"], help="PDF, DOCX or TXT")
uploaded_text = ""
if uploaded is not None:
    try:
        uploaded_text = extract_uploaded_text(uploaded).strip()
        st.success(f"Extracted text from {uploaded.name}")
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")

profile_text = st.text_area(
    "Profile or skills",
    value=uploaded_text,
    height=190,
    placeholder="Example: Python, SQL, pandas, NumPy, machine learning, TensorFlow...",
    help="Review the extracted text or enter a skill profile manually.",
)

left, right = st.columns(2)
with left:
    top_k = st.slider("Number of career recommendations", min_value=3, max_value=10, value=5)
with right:
    target_career = st.text_input("Target career (optional)", placeholder="Example: Data Scientist")

if st.button("Analyse Career Profile", type="primary", use_container_width=True):
    if not profile_text.strip():
        st.warning("Enter profile text or upload a resume first.")
    else:
        try:
            with st.spinner("Running CareerCast models and skill-gap analysis..."):
                prediction = api_post("/predict", {"skills_text": profile_text, "top_k": top_k})
                recommendation = api_post("/recommend", {"skills_text": profile_text, "top_k": top_k})
                gap_payload: dict[str, Any] = {"skills_text": profile_text, "top_k_careers": 1}
                if target_career.strip():
                    gap_payload["target_career"] = target_career.strip()
                gap_report = api_post("/gap-report", gap_payload)
                model_info = api_get("/models/info")
            st.session_state["career_result"] = {
                "profile_text": profile_text,
                "prediction": prediction,
                "recommendation": recommendation,
                "gap_report": gap_report,
                "model_info": model_info,
            }
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Analysis failed: {exc}")

result = st.session_state.get("career_result")
if result:
    prediction = result["prediction"]
    recommendation = result["recommendation"]
    gap_report = result["gap_report"]
    model_info = result["model_info"]
    top_predictions = prediction.get("top_predictions", [])
    recommendations = recommendation.get("recommendations", [])
    gaps = gap_report.get("gap_analysis", [])
    primary_gap = gaps[0] if gaps else {}
    primary = top_predictions[0] if top_predictions else {}

    st.markdown("## Career Intelligence Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted career", primary.get("career", "Unavailable"))
    c2.metric("Prediction score", f"{float(primary.get('probability', 0)) * 100:.2f}%")
    c3.metric("Skill alignment", f"{float(primary_gap.get('alignment_score', 0)):.2f}%")
    c4.metric("Prediction model", primary.get("model", "Unavailable"))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Top Careers", "Skill Gap", "Career Comparison", "Model Review", "Download Report"]
    )
    with tab1:
        if recommendations:
            st.plotly_chart(probability_chart(recommendations), use_container_width=True)
            display = pd.DataFrame(recommendations).rename(columns={
                "rank": "Rank", "career": "Career", "ensemble_score": "Ensemble score",
                "lr_probability": "LR", "rf_probability": "RF", "xgb_probability": "XGBoost",
            })
            st.dataframe(display, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(f"### Target: {gap_report.get('target_career', 'Unavailable')}")
        matched = primary_gap.get("matched_skills", [])
        st.markdown("**Matched skills**")
        if matched:
            st.markdown("".join(f'<span class="chip">{skill}</span>' for skill in matched), unsafe_allow_html=True)
        else:
            st.info("No matched skills were detected for this target.")

        st.markdown("### Missing skills and actions")
        missing = primary_gap.get("missing_skills", [])
        if missing:
            gap_frame = pd.DataFrame(missing).rename(columns={
                "skill": "Skill", "weight": "Weight", "priority": "Priority", "suggestion": "Action",
            })
            st.dataframe(gap_frame, use_container_width=True, hide_index=True)
            summary = primary_gap.get("priority_summary", {})
            p1, p2, p3 = st.columns(3)
            p1.metric("High priority", summary.get("High", 0))
            p2.metric("Medium priority", summary.get("Medium", 0))
            p3.metric("Low priority", summary.get("Low", 0))
        else:
            st.success("No missing skills were identified.")

    with tab3:
        st.markdown("### Compare recommended careers")
        career_options = [item["career"] for item in recommendations]
        if len(career_options) < 2:
            st.info("At least two recommendations are required for comparison.")
        else:
            left_choice, right_choice = st.columns(2)
            with left_choice:
                career_a = st.selectbox("Career A", career_options, index=0)
            with right_choice:
                career_b = st.selectbox("Career B", career_options, index=1)
            if career_a == career_b:
                st.warning("Select two different careers.")
            elif st.button("Compare Careers", use_container_width=True):
                try:
                    with st.spinner("Comparing career skill requirements..."):
                        comparison = []
                        for career in (career_a, career_b):
                            gap = api_post(
                                "/gap-report",
                                {
                                    "skills_text": result["profile_text"],
                                    "target_career": career,
                                    "top_k_careers": 1,
                                },
                            )["gap_analysis"][0]
                            comparison.append(gap)
                    st.session_state["career_comparison"] = comparison
                except (requests.RequestException, RuntimeError, KeyError, IndexError) as exc:
                    st.error(f"Career comparison failed: {exc}")

            comparison = st.session_state.get("career_comparison")
            if comparison:
                comparison_frame = pd.DataFrame(
                    [
                        {
                            "Career": item["career"],
                            "Skill alignment (%)": item["alignment_score"],
                            "Matched skills": len(item["matched_skills"]),
                            "Missing skills": len(item["missing_skills"]),
                            "High-priority gaps": item["priority_summary"].get("High", 0),
                        }
                        for item in comparison
                    ]
                )
                st.plotly_chart(
                    px.bar(
                        comparison_frame,
                        x="Career",
                        y="Skill alignment (%)",
                        color="Career",
                        range_y=[0, 100],
                        title="Skill alignment comparison",
                    ),
                    use_container_width=True,
                )
                st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
                columns = st.columns(2)
                for column, item in zip(columns, comparison):
                    with column:
                        st.markdown(f"#### {item['career']}")
                        missing_names = [entry["skill"] for entry in item["missing_skills"][:10]]
                        st.write("Top missing skills:", ", ".join(missing_names) or "None")

    with tab4:
        st.json({
            "embedding_model": model_info.get("embedding_model"),
            "embedding_dimension": model_info.get("embedding_dimension"),
            "career_classes": model_info.get("n_classes"),
            "classifiers": model_info.get("classifiers"),
            "ensemble_weights": recommendation.get("weights_used"),
        })
        st.caption("The selected prediction model and metrics come from verified Milestone 2 artifacts; models are not retrained by this interface.")

    with tab5:
        try:
            pdf_bytes = build_career_report(
                result["profile_text"], prediction, recommendation, gap_report, model_info
            )
            st.download_button(
                "Download CareerCast PDF Report",
                data=pdf_bytes,
                file_name="CareerCast_Career_Report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
            st.caption("The PDF contains the current prediction, recommendations, weighted gaps, priorities, actions, model information and timestamp.")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

st.markdown("---")
st.caption("CareerCast | Milestone 4 | Decision-support prototype")

"""Milestone 2 analytics view built from verified artifacts and live API output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSNE_PATH = PROJECT_ROOT / "streamlit_app/assets/finetuned_sbert_tsne.csv"
SEMEVAL_PATH = PROJECT_ROOT / "results/milestone2_semeval/semeval_sts_results.json"
LINKEDIN_PATH = PROJECT_ROOT / "results/milestone2_linkedin/linkedin_transition_validation_summary.json"


def build_model_metrics_frame(metrics_summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    for key, values in metrics_summary.get("all_models", {}).items():
        rows.append(
            {
                "Model": display_names.get(key, key.replace("_", " ").title()),
                "CV accuracy": float(values.get("best_cv_accuracy", 0)),
                "Test accuracy": float(values.get("test_accuracy", 0)),
                "Macro F1": float(values.get("macro_f1", 0)),
                "Top-3 accuracy": float(values.get("top3_accuracy", 0)),
                "Top-5 accuracy": float(values.get("top5_accuracy", 0)),
            }
        )
    return pd.DataFrame(rows)


def load_tsne_projection(path: Path = TSNE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"career", "tsne_x", "tsne_y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"t-SNE CSV is missing columns: {sorted(missing)}")
    return frame


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_milestone2_dashboard(
    api_get: Callable[[str], dict[str, Any]],
    api_post: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    st.markdown("## Milestone 2 – Advanced ML & Recommendation Engine")
    st.caption("Verified model evaluation, live Top-K ranking and fine-tuned SBERT embedding analysis.")

    model_info = api_get("/models/info")
    summary = model_info.get("metrics_summary") or {}
    metrics = build_model_metrics_frame(summary)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active pipeline", model_info.get("pipeline", "Unavailable"))
    m2.metric("Fine-tuned SBERT", "Yes" if model_info.get("embedding_fine_tuned") else "No")
    m3.metric("Embedding dimension", model_info.get("embedding_dimension", "—"))
    m4.metric("Career classes", model_info.get("n_classes", "—"))

    st.markdown("### Model comparison – Macro F1")
    if metrics.empty:
        st.warning("Verified classifier metrics are unavailable from /models/info.")
    else:
        chart = px.bar(
            metrics,
            x="Model",
            y="Macro F1",
            color="Model",
            text=metrics["Macro F1"].map(lambda value: f"{value:.4f}"),
            range_y=[0, 1.02],
            color_discrete_sequence=["#4f46e5", "#0f766e", "#d97706"],
        )
        chart.add_hline(y=0.80, line_dash="dash", line_color="#dc2626", annotation_text="Required threshold: 0.80")
        chart.update_layout(showlegend=False, height=390, margin=dict(l=10, r=20, t=20, b=10))
        chart.update_traces(textposition="outside")
        st.plotly_chart(chart, use_container_width=True)
        formatted = metrics.copy()
        for column in formatted.columns[1:]:
            formatted[column] = formatted[column].map(lambda value: f"{value * 100:.3f}%")
        st.dataframe(formatted, use_container_width=True, hide_index=True)
        st.caption("These are held-out aggregate evaluation metrics, not confidence guarantees for every resume.")

    st.markdown("### Live Top-5 career recommendations")
    profile = st.text_area(
        "Skills or profile text",
        value="Python, SQL, pandas, NumPy, scikit-learn, machine learning, statistics, data visualization, TensorFlow, deep learning",
        height=120,
        key="m2_profile_text",
    )
    if st.button("Generate Top-5 Recommendations", type="primary", use_container_width=True, key="m2_recommend"):
        if not profile.strip():
            st.warning("Enter skills or profile text first.")
        else:
            with st.spinner("Running all three classifiers..."):
                recommendation = api_post("/recommend", {"skills_text": profile, "top_k": 5})
                top = recommendation.get("recommendations", [])
                gap = None
                if top:
                    gap = api_post(
                        "/gap-report",
                        {"skills_text": profile, "target_career": top[0]["career"], "top_k_careers": 1},
                    )
            st.session_state["m2_live_result"] = {"recommendation": recommendation, "gap": gap}

    live = st.session_state.get("m2_live_result")
    if live:
        recommendations = live["recommendation"].get("recommendations", [])
        if recommendations:
            frame = pd.DataFrame(recommendations).rename(
                columns={
                    "rank": "Rank", "career": "Career", "ensemble_score": "Ensemble",
                    "lr_probability": "Logistic Regression", "rf_probability": "Random Forest",
                    "xgb_probability": "XGBoost",
                }
            )
            plot_frame = frame.copy()
            plot_frame["Confidence (%)"] = plot_frame["Ensemble"] * 100
            rank_chart = px.bar(
                plot_frame.sort_values("Confidence (%)"), x="Confidence (%)", y="Career",
                orientation="h", color="Confidence (%)", color_continuous_scale=["#0f766e", "#4f46e5"],
                text=plot_frame.sort_values("Confidence (%)")["Confidence (%)"].map(lambda value: f"{value:.2f}%"),
            )
            rank_chart.update_layout(coloraxis_showscale=False, height=360, margin=dict(l=10, r=20, t=20, b=10))
            rank_chart.update_traces(textposition="outside")
            st.plotly_chart(rank_chart, use_container_width=True)
            st.dataframe(frame, use_container_width=True, hide_index=True)
            gap_rows = (live.get("gap") or {}).get("gap_analysis", [])
            if gap_rows:
                st.metric("Top-career skill alignment", f"{float(gap_rows[0].get('alignment_score', 0)):.2f}%")

    st.markdown("### Fine-tuned SBERT t-SNE projection")
    try:
        projection = load_tsne_projection()
        careers = sorted(projection["career"].dropna().unique().tolist())
        default = careers[: min(8, len(careers))]
        selected = st.multiselect("Careers displayed", careers, default=default, key="m2_tsne_careers")
        visible = projection[projection["career"].isin(selected)] if selected else projection
        scatter = px.scatter(
            visible, x="tsne_x", y="tsne_y", color="career", hover_name="career",
            labels={"tsne_x": "t-SNE component 1", "tsne_y": "t-SNE component 2", "career": "Career"},
            opacity=0.78,
        )
        scatter.update_traces(marker={"size": 7})
        scatter.update_layout(height=540, margin=dict(l=10, r=20, t=20, b=10))
        st.plotly_chart(scatter, use_container_width=True)
        st.caption(f"Balanced projection: {len(projection)} samples across {projection['career'].nunique()} careers. t-SNE is for visual exploration, not an accuracy metric.")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.info(f"Generate the t-SNE visualization CSV before deployment: {exc}")

    st.markdown("### External validation evidence")
    semeval = _read_json(SEMEVAL_PATH)
    linkedin = _read_json(LINKEDIN_PATH)
    left, right = st.columns(2)
    with left:
        st.markdown("#### SemEval-2017 STS")
        if semeval and semeval.get("results"):
            result = next((item for item in semeval["results"] if item.get("model") == "finetuned_career_sbert"), semeval["results"][-1])
            st.metric("Pearson correlation", f"{float(result.get('pearson_r', 0)):.4f}")
            st.metric("Spearman correlation", f"{float(result.get('spearman_rho', 0)):.4f}")
            st.caption(f"{semeval.get('benchmark', 'SemEval benchmark')} · {semeval.get('n_pairs', 0)} pairs")
        else:
            st.info("SemEval result artifact is unavailable.")
    with right:
        st.markdown("#### LinkedIn transitions")
        if linkedin:
            st.metric("Mean Pearson ranking correlation", f"{float(linkedin.get('mean_pearson_ranking_correlation', 0)):.4f}")
            st.metric("Mean Spearman ranking correlation", f"{float(linkedin.get('mean_spearman_ranking_correlation', 0)):.4f}")
            st.caption(f"{linkedin.get('transition_pairs', 0)} transition pairs across {linkedin.get('source_careers', 0)} source careers")
        else:
            st.info("LinkedIn transition result artifact is unavailable.")
    st.caption("LinkedIn statistics describe consistency within the curated transition rankings; they are not direct classifier accuracy.")

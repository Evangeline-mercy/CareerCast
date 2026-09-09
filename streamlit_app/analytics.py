"""Pure data preparation helpers for CareerCast cohort analytics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


MAX_COHORT_SIZE = 50


def confidence_level(score_percent: float) -> str:
    """Return a plain-language band for an ensemble score expressed as a percentage."""
    if score_percent >= 60:
        return "High"
    if score_percent >= 35:
        return "Moderate"
    return "Low"


def build_cohort_result_row(
    record: dict[str, Any],
    recommendations: list[dict[str, Any]],
    gap: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build one detailed, CSV-safe cohort result row from live API responses."""
    if not recommendations:
        raise ValueError("At least one career recommendation is required.")
    top = recommendations[0]
    score = round(float(top.get("ensemble_score", 0.0)) * 100, 2)
    matched = gap.get("matched_skills", [])
    missing = gap.get("missing_skills", [])
    high_priority = [
        str(item.get("skill", "")).strip()
        for item in missing
        if isinstance(item, dict) and str(item.get("priority", "")).lower() == "high"
    ]
    ranked = " | ".join(
        f"{item.get('career', 'Unavailable')} ({float(item.get('ensemble_score', 0.0)) * 100:.2f}%)"
        for item in recommendations[:3]
    )
    return {
        "Name": record.get("name", ""),
        "Input skills": record.get("skills", ""),
        "Predicted career": top.get("career", "Unavailable"),
        "Career score (%)": score,
        "Confidence level": confidence_level(score),
        "Top 3 recommendations": ranked,
        "Skill alignment (%)": round(float(gap.get("alignment_score", 0.0)), 2),
        "Matched skills": ", ".join(str(item) for item in matched) or "None",
        "High-priority gaps": len(high_priority),
        "High-priority missing skills": ", ".join(high_priority) or "None",
        "Generated at (UTC)": generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def prepare_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an uploaded cohort table."""
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    if "skills" not in normalized.columns:
        raise ValueError("The cohort CSV must contain a 'skills' column.")
    if "name" not in normalized.columns:
        normalized.insert(0, "name", [f"Profile {index}" for index in range(1, len(normalized) + 1)])
    normalized = normalized[["name", "skills"]].fillna("")
    normalized["name"] = normalized["name"].astype(str).str.strip()
    normalized["skills"] = normalized["skills"].astype(str).str.strip()
    normalized = normalized[normalized["skills"] != ""].reset_index(drop=True)
    if normalized.empty:
        raise ValueError("The cohort CSV does not contain any non-empty skill profiles.")
    if len(normalized) > MAX_COHORT_SIZE:
        raise ValueError(f"A cohort can contain at most {MAX_COHORT_SIZE} profiles per run.")
    normalized.loc[normalized["name"] == "", "name"] = [
        f"Profile {index}" for index in normalized.index[normalized["name"] == ""] + 1
    ]
    return normalized


def summarize_cohort(result: pd.DataFrame) -> dict[str, Any]:
    """Calculate deterministic headline metrics and career distribution."""
    required = {"Predicted career", "Skill alignment (%)"}
    if not required.issubset(result.columns):
        raise ValueError("Cohort results are missing required analytics columns.")
    distribution = (
        result["Predicted career"]
        .value_counts()
        .rename_axis("Predicted career")
        .reset_index(name="Profiles")
    )
    return {
        "profile_count": int(len(result)),
        "career_count": int(result["Predicted career"].nunique()),
        "mean_alignment": float(result["Skill alignment (%)"].mean()),
        "career_distribution": distribution,
    }

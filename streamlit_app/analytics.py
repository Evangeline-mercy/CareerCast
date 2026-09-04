"""Pure data preparation helpers for CareerCast cohort analytics."""

from __future__ import annotations

from typing import Any

import pandas as pd


MAX_COHORT_SIZE = 50


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

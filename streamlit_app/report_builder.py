"""Generate a CareerCast PDF report from live API results."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape


_FONT_DIR = Path(reportlab.__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont("CareerSans", _FONT_DIR / "Vera.ttf"))
pdfmetrics.registerFont(TTFont("CareerSansBold", _FONT_DIR / "VeraBd.ttf"))


def _text(value: Any) -> str:
    return escape(str(value if value is not None else ""))


def _skill_names(items: list[Any]) -> str:
    names = [item.get("skill", "") if isinstance(item, dict) else str(item) for item in items]
    return ", ".join(name for name in names if name) or "None"


def build_career_report(
    profile_text: str,
    prediction: dict[str, Any],
    recommendation: dict[str, Any],
    gap_report: dict[str, Any],
    model_info: dict[str, Any],
) -> bytes:
    """Return a PDF containing the current, non-hard-coded CareerCast result."""
    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="CareerCast Career Intelligence Report",
        author="CareerCast",
    )
    styles = getSampleStyleSheet()
    styles["BodyText"].fontName = "CareerSans"
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], fontName="CareerSansBold", alignment=TA_CENTER, textColor=colors.HexColor("#4338CA")))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Heading3"], fontName="CareerSansBold", textColor=colors.HexColor("#111827")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="CareerSansBold", textColor=colors.HexColor("#312E81"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="CareerSans", fontSize=8.5, leading=11))

    predictions = prediction.get("top_predictions", [])
    recommendations = recommendation.get("recommendations", [])
    gaps = gap_report.get("gap_analysis", [])
    primary_gap = gaps[0] if gaps else {}
    top_prediction = predictions[0] if predictions else {}
    target = gap_report.get("target_career") or top_prediction.get("career", "Unavailable")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story = [
        Paragraph("CareerCast", styles["TitleCenter"]),
        Paragraph("AI-Powered Career Path Prediction & Skill Intelligence", styles["Subtitle"]),
        Spacer(1, 6),
        Paragraph(f"Generated: {_text(generated)}", styles["Small"]),
        Paragraph("Profile Summary", styles["Section"]),
        Paragraph(_text(profile_text[:1200]), styles["BodyText"]),
        Paragraph("Primary Prediction", styles["Section"]),
    ]

    probability = float(top_prediction.get("probability", 0.0))
    primary_data = [
        ["Career", "Probability", "Model", "Skill alignment"],
        [
            _text(target),
            f"{probability * 100:.2f}%",
            _text(top_prediction.get("model", "Unavailable")),
            f"{float(primary_gap.get('alignment_score', 0.0)):.2f}%",
        ],
    ]
    primary_table = Table(primary_data, colWidths=[65 * mm, 32 * mm, 45 * mm, 32 * mm])
    primary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "CareerSansBold"),
        ("FONTNAME", (0, 1), (-1, -1), "CareerSans"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.extend([primary_table, Paragraph("Top-K Careers", styles["Section"])])

    ranking_rows = [["Rank", "Career", "Ensemble score", "LR", "RF", "XGB"]]
    for item in recommendations:
        ranking_rows.append([
            item.get("rank", ""),
            Paragraph(_text(item.get("career", "")), styles["Small"]),
            f"{float(item.get('ensemble_score', 0)):.4f}",
            f"{float(item.get('lr_probability', 0)):.4f}",
            f"{float(item.get('rf_probability', 0)):.4f}",
            f"{float(item.get('xgb_probability', 0)):.4f}",
        ])
    ranking_table = Table(ranking_rows, repeatRows=1, colWidths=[13 * mm, 61 * mm, 27 * mm, 23 * mm, 23 * mm, 23 * mm])
    ranking_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (-1, 0), "CareerSansBold"),
        ("FONTNAME", (0, 1), (-1, -1), "CareerSans"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([
        ranking_table,
        Paragraph("Skill Gap Analysis", styles["Section"]),
        Paragraph(f"<b>Matched skills:</b> {_text(_skill_names(primary_gap.get('matched_skills', [])))}", styles["BodyText"]),
        Spacer(1, 5),
    ])

    priority_summary = primary_gap.get("priority_summary", {})
    story.extend([
        Paragraph(
            "<b>Priority summary:</b> "
            f"High {_text(priority_summary.get('High', 0))} | "
            f"Medium {_text(priority_summary.get('Medium', 0))} | "
            f"Low {_text(priority_summary.get('Low', 0))}",
            styles["BodyText"],
        ),
        Spacer(1, 6),
    ])

    missing_rows = [["Missing skill", "Weight", "Priority", "Actionable suggestion"]]
    for item in primary_gap.get("missing_skills", []):
        missing_rows.append([
            Paragraph(_text(item.get("skill", "")), styles["Small"]),
            f"{float(item.get('weight', 0)):.4f}",
            _text(item.get("priority", "")),
            Paragraph(_text(item.get("suggestion", "")), styles["Small"]),
        ])
    if len(missing_rows) == 1:
        missing_rows.append(["None", "-", "-", "No missing skills identified."])
    missing_table = Table(missing_rows, repeatRows=1, colWidths=[35 * mm, 19 * mm, 22 * mm, 94 * mm])
    missing_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#B45309")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (-1, 0), "CareerSansBold"),
        ("FONTNAME", (0, 1), (-1, -1), "CareerSans"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([
        missing_table,
        Spacer(1, 10),
        Paragraph("Model Information", styles["Section"]),
        Paragraph(f"<b>Embedding model:</b> {_text(model_info.get('embedding_model', 'Unavailable'))}", styles["BodyText"]),
        Paragraph(f"<b>Embedding dimension:</b> {_text(model_info.get('embedding_dimension', 'Unavailable'))}", styles["BodyText"]),
        Paragraph(f"<b>Career classes:</b> {_text(model_info.get('n_classes', 'Unavailable'))}", styles["BodyText"]),
        Paragraph(f"<b>Classifiers:</b> {_text(', '.join(model_info.get('classifiers', [])))}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("This report is generated from the current user input and live CareerCast API responses. Model probabilities are decision-support scores, not guaranteed career outcomes.", styles["Small"]),
    ])

    doc.build(story)
    return output.getvalue()

import json
import os
import inspect
from pathlib import Path

import pandas as pd
import numpy as np
import joblib


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_read_json(path):
    if not Path(path).exists():
        print(f"  MISSING: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"  FOUND: {path}")
        return data

    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None


def safe_read_csv(path, nrows=None):
    if not Path(path).exists():
        print(f"  MISSING: {path}")
        return None

    try:
        df = pd.read_csv(path, nrows=nrows)
        print(f"  FOUND: {path}  shape={df.shape}")
        return df

    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None


# ================================================================
# A. RANDOM FOREST
# ================================================================

section("A. RANDOM FOREST")

rf_meta = safe_read_json(
    "results/tree_model/metadata.json"
)

if rf_meta:
    print("\nMetadata:")
    print(json.dumps(rf_meta, indent=2))


rf_cv = safe_read_csv(
    "results/tree_model/cv_results.csv"
)

if rf_cv is not None:

    print("\nCV columns:")
    print(list(rf_cv.columns))

    print("\nFirst 10 rows:")
    print(rf_cv.head(10).to_string())

    param_cols = [
        c for c in rf_cv.columns
        if c.startswith("param_")
    ]

    print("\nHyperparameters searched:")
    print(param_cols)

    if "mean_test_score" in rf_cv.columns:

        best_row = rf_cv.loc[
            rf_cv["mean_test_score"].idxmax()
        ]

        print(
            "\nBest CV score:",
            best_row["mean_test_score"]
        )

        print(
            "Best parameters:"
        )

        for c in param_cols:
            print(
                f"  {c}: {best_row[c]}"
            )


rf_report = safe_read_json(
    "results/tree_model/classification_report.json"
)

if rf_report:

    print("\nClassification report keys:")
    print(list(rf_report.keys()))

    if "accuracy" in rf_report:
        print(
            "Accuracy:",
            rf_report["accuracy"]
        )


rf_encoder_path = (
    "results/tree_model/label_encoder.joblib"
)

if Path(rf_encoder_path).exists():

    try:
        rf_le = joblib.load(rf_encoder_path)

        print(
            "\nRF number of classes:",
            len(rf_le.classes_)
        )

        print(
            "First 20 RF classes:"
        )

        print(
            list(rf_le.classes_[:20])
        )

    except Exception as e:
        rf_le = None
        print(
            "Could not load RF label encoder:",
            e
        )

else:

    rf_le = None
    print(
        "\nMISSING:",
        rf_encoder_path
    )


# ================================================================
# B. XGBOOST
# ================================================================

section("B. XGBOOST")

xgb_meta = safe_read_json(
    "results/xgboost_model/metadata.json"
)

if xgb_meta:
    print("\nMetadata:")
    print(json.dumps(xgb_meta, indent=2))


xgb_cv = safe_read_csv(
    "results/xgboost_model/cv_results.csv"
)

if xgb_cv is not None:

    print("\nCV columns:")
    print(list(xgb_cv.columns))

    print("\nFirst 10 rows:")
    print(xgb_cv.head(10).to_string())

    param_cols = [
        c for c in xgb_cv.columns
        if c.startswith("param_")
    ]

    print("\nHyperparameters searched:")
    print(param_cols)

    if "mean_test_score" in xgb_cv.columns:

        best_row = xgb_cv.loc[
            xgb_cv["mean_test_score"].idxmax()
        ]

        print(
            "\nBest CV score:",
            best_row["mean_test_score"]
        )

        print(
            "Best parameters:"
        )

        for c in param_cols:
            print(
                f"  {c}: {best_row[c]}"
            )


xgb_report = safe_read_json(
    "results/xgboost_model/classification_report.json"
)

if xgb_report:

    print("\nClassification report keys:")
    print(list(xgb_report.keys()))

    if "accuracy" in xgb_report:
        print(
            "Accuracy:",
            xgb_report["accuracy"]
        )


xgb_encoder_path = (
    "results/xgboost_model/label_encoder.joblib"
)

if Path(xgb_encoder_path).exists():

    try:
        xgb_le = joblib.load(xgb_encoder_path)

        print(
            "\nXGBoost number of classes:",
            len(xgb_le.classes_)
        )

        print(
            "First 20 XGBoost classes:"
        )

        print(
            list(xgb_le.classes_[:20])
        )

    except Exception as e:

        xgb_le = None

        print(
            "Could not load XGBoost label encoder:",
            e
        )

else:

    xgb_le = None

    print(
        "\nMISSING:",
        xgb_encoder_path
    )


# ================================================================
# C. DATASET / CAREER ALIGNMENT
# ================================================================

section("C. CAREER DATASET")

career_df = safe_read_csv(
    "results/careercast_candidate_profiles.csv"
)

if career_df is not None:

    print("\nDataset columns:")
    print(list(career_df.columns))

    if "Career" in career_df.columns:

        print(
            "\nUnique careers:",
            career_df["Career"].nunique()
        )

        print(
            "Total rows:",
            len(career_df)
        )

        print(
            "\nFirst 20 careers:"
        )

        print(
            career_df["Career"]
            .dropna()
            .unique()[:20]
        )

    else:

        print(
            "\nWARNING: 'Career' column not found."
        )


if rf_le is not None and xgb_le is not None:

    rf_set = set(rf_le.classes_)
    xgb_set = set(xgb_le.classes_)

    print("\nRF vs XGBoost label comparison:")

    if rf_set == xgb_set:

        print(
            "  MATCH: Both models contain exactly the same career classes."
        )

    else:

        print(
            "  MISMATCH detected."
        )

        print(
            "  Only RF:",
            list(rf_set - xgb_set)[:20]
        )

        print(
            "  Only XGBoost:",
            list(xgb_set - rf_set)[:20]
        )


# ================================================================
# D. SBERT
# ================================================================

section("D. SBERT")

sbert_dir = Path(
    "results/semantic_embeddings/sbert_finetuned"
)

if sbert_dir.exists():

    print(
        "\nSBERT directory FOUND:"
    )

    for p in sbert_dir.rglob("*"):

        if p.is_file():

            print(
                f"  {p.relative_to(sbert_dir)} "
                f"({p.stat().st_size} bytes)"
            )

    config_path = sbert_dir / "config.json"

    if config_path.exists():

        try:

            with open(
                config_path,
                "r",
                encoding="utf-8"
            ) as f:

                cfg = json.load(f)

            print(
                "\nSBERT config:"
            )

            print(
                json.dumps(
                    cfg,
                    indent=2
                )[:3000]
            )

        except Exception as e:

            print(
                "Could not read SBERT config:",
                e
            )

else:

    print(
        "\nMISSING SBERT directory:",
        sbert_dir
    )


print("\nChecking fine-tuning evidence:")

evidence_files = [
    "README.md",
    "train_script.py",
    "training_args.json",
    "modules.json",
    "sentence_bert_config.json"
]

for name in evidence_files:

    path = sbert_dir / name

    if path.exists():
        print(
            f"  FOUND: {name}"
        )
    else:
        print(
            f"  NOT FOUND: {name}"
        )


comparison_path = (
    "results/semantic_embeddings/"
    "sbert_old_vs_finetuned_comparison.csv"
)

comparison_df = safe_read_csv(
    comparison_path
)

if comparison_df is not None:

    print(
        "\nSBERT comparison columns:"
    )

    print(
        list(comparison_df.columns)
    )

    print(
        "\nComparison data:"
    )

    print(
        comparison_df.head(10).to_string()
    )


emb_path = (
    "results/semantic_embeddings/"
    "finetuned_career_embeddings.npy"
)

label_path = (
    "results/semantic_embeddings/"
    "finetuned_career_labels.csv"
)

if Path(emb_path).exists():

    emb = np.load(emb_path)

    print(
        "\nEmbedding shape:",
        emb.shape
    )

else:

    print(
        "\nMISSING:",
        emb_path
    )

if Path(label_path).exists():

    labels = pd.read_csv(label_path)

    print(
        "Embedding labels rows:",
        len(labels)
    )

else:

    print(
        "MISSING:",
        label_path
    )


# ================================================================
# E. RECOMMENDER SOURCE
# ================================================================

section("E. CAREER RECOMMENDER SOURCE")

try:

    import career_recommender_v4 as v4

    print(
        "\ncareer_recommender_v4.py imported successfully."
    )

    functions = [
        "skill_similarity",
        "calculate_skill_match",
        "robust_normalize",
        "calculate_recommendation_confidence",
        "calculate_confidence",
        "get_topk",
        "recommend"
    ]

    for name in functions:

        fn = getattr(
            v4,
            name,
            None
        )

        if fn is not None and callable(fn):

            print(
                f"\n--- {name}() ---"
            )

            try:

                print(
                    inspect.getsource(fn)
                )

            except Exception as e:

                print(
                    "Could not retrieve source:",
                    e
                )

        else:

            print(
                f"\n--- {name}() NOT FOUND ---"
            )

except Exception as e:

    print(
        "\nCould not import career_recommender_v4:"
    )

    print(
        repr(e)
    )


# ================================================================
# F. LATEST RECOMMENDATION OUTPUT
# ================================================================

section("F. LATEST V4 OUTPUT")

output_dir = Path("v4_output")

if output_dir.exists():

    runs = sorted(
        [
            d for d in output_dir.iterdir()
            if d.is_dir()
        ]
    )

    if runs:

        latest = runs[-1]

        print(
            "\nLatest run:",
            latest
        )

        rec_file = (
            latest /
            "v4_recommendations.csv"
        )

        if rec_file.exists():

            df = pd.read_csv(
                rec_file
            )

            print(
                "\nColumns:"
            )

            print(
                list(df.columns)
            )

            print(
                "\nRows:",
                len(df)
            )

            print(
                "\nFirst 15 recommendations:"
            )

            print(
                df.head(15).to_string(
                    index=False
                )
            )

            if "confidence_percentage" in df.columns:

                print(
                    "\nConfidence statistics:"
                )

                print(
                    df[
                        "confidence_percentage"
                    ].describe()
                )

                print(
                    "\nUnique confidence values:"
                )

                print(
                    df[
                        "confidence_percentage"
                    ].unique()
                )

            if "skill_match" in df.columns:

                print(
                    "\nSkill match statistics:"
                )

                print(
                    df[
                        "skill_match"
                    ].describe()
                )

            if "skill_alignment" in df.columns:

                print(
                    "\nSkill alignment statistics:"
                )

                print(
                    df[
                        "skill_alignment"
                    ].describe()
                )

        else:

            print(
                "\nMissing:",
                rec_file
            )

    else:

        print(
            "\nNo run directories found."
        )

else:

    print(
        "\nMISSING v4_output directory."
    )


# ================================================================
# END
# ================================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CareerCast Milestone 2
# Candidate Profile Dataset Generator
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE = BASE_DIR / "results" / "careercast_milestone2_dataset.csv"
OUTPUT = BASE_DIR / "results" / "careercast_candidate_profiles.csv"

RANDOM_STATE = 42
SAMPLES_PER_CAREER = 20

rng = np.random.default_rng(RANDOM_STATE)

print("=" * 75)
print("CAREERCAST MILESTONE 2 - CANDIDATE PROFILE GENERATION")
print("=" * 75)

# ------------------------------------------------------------
# Load source dataset
# ------------------------------------------------------------

df = pd.read_csv(SOURCE)

print(f"\nSource rows: {len(df)}")
print(f"Source careers: {df['Career'].nunique()}")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def split_items(text):
    if pd.isna(text):
        return []

    return [
        x.strip()
        for x in str(text).split(",")
        if x.strip()
    ]


def make_skill_subset(skills, keep_ratio):
    """
    Create a candidate skill profile by selecting a subset
    of the career's available software skills.
    """

    if not skills:
        return ""

    n = len(skills)

    keep = max(
        1,
        int(round(n * keep_ratio))
    )

    keep = min(keep, n)

    selected = rng.choice(
        skills,
        size=keep,
        replace=False
    )

    return ", ".join(selected)


def make_experience(base):
    """
    Generate realistic variation around the source
    occupation experience value.
    """

    value = float(base)

    variation = rng.normal(
        loc=0.0,
        scale=1.5
    )

    result = max(
        0.0,
        value + variation
    )

    return round(result, 1)


def make_education(base):
    """
    Keep the source education signal but introduce
    small candidate-level variation.
    """

    value = float(base)

    variation = rng.normal(
        loc=0.0,
        scale=0.15
    )

    result = np.clip(
        value + variation,
        1.0,
        12.0
    )

    return round(result, 2)


# ------------------------------------------------------------
# Generate candidate profiles
# ------------------------------------------------------------

records = []

for _, row in df.iterrows():

    career = row["Career"]

    software_skills = split_items(
        row["Software_Skills"]
    )

    # --------------------------------------------------------
    # Generate multiple candidate profiles
    # --------------------------------------------------------

    for profile_id in range(
        SAMPLES_PER_CAREER
    ):

        # Different skill completeness levels
        keep_ratio = rng.uniform(
            0.55,
            1.00
        )

        candidate_skills = make_skill_subset(
            software_skills,
            keep_ratio
        )

        # Add a small probability of retaining
        # the full software profile.
        if rng.random() < 0.15:
            candidate_skills = ", ".join(
                software_skills
            )

        records.append({

            "Profile_ID":
                f"{career}_{profile_id+1}",

            "Career":
                career,

            "Skills":
                candidate_skills,

            "Education":
                make_education(
                    row["Education_Level"]
                ),

            "Experience":
                make_experience(
                    row["Experience_Level"]
                ),

            "Realistic":
                row["Realistic"],

            "Investigative":
                row["Investigative"],

            "Artistic":
                row["Artistic"],

            "Social":
                row["Social"],

            "Enterprising":
                row["Enterprising"],

            "Conventional":
                row["Conventional"],

            "Job_Description":
                row["Job_Description"]

        })


# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

profiles = pd.DataFrame(records)

# ------------------------------------------------------------
# Shuffle
# ------------------------------------------------------------

profiles = profiles.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

profiles.to_csv(
    OUTPUT,
    index=False
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("GENERATED DATASET")
print("=" * 75)

print(
    "Rows:",
    len(profiles)
)

print(
    "Columns:",
    len(profiles.columns)
)

print(
    "Unique careers:",
    profiles["Career"].nunique()
)

print(
    "Samples per career:",
    profiles["Career"].value_counts().describe()
)

print(
    "\nMissing values:"
)

print(
    profiles.isnull().sum().to_string()
)

print(
    "\nCareer distribution:"
)

print(
    profiles["Career"]
    .value_counts()
    .head(10)
    .to_string()
)

print(
    "\nSaved to:"
)

print(OUTPUT)

print(
    "\n" + "=" * 75
)
print(
    "PROFILE GENERATION COMPLETE"
)
print(
    "=" * 75
)
# ============================================================
# CAREERCAST MILESTONE 2
# TASK 2 - SENTENCE-BERT FINE-TUNING
#
# Fine-tunes SBERT using candidate/profile text and
# corresponding job descriptions.
#
# The Career label is NOT included in the input text.
# This prevents direct career-label leakage.
# ============================================================

import os
import pandas as pd

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses
)

from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "results/careercast_candidate_profiles.csv"

BASE_MODEL = "all-MiniLM-L6-v2"

OUTPUT_DIR = (
    "results/semantic_embeddings/"
    "sbert_finetuned"
)

BATCH_SIZE = 16
EPOCHS = 2
WARMUP_RATIO = 0.1


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("CAREERCAST MILESTONE 2")
print("TASK 2 - SENTENCE-BERT FINE-TUNING")
print("=" * 75)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset rows :", len(df))
print("Unique careers:", df["Career"].nunique())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Career",
    "Skills",
    "Education",
    "Experience",
    "Job_Description"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Required columns: PASS")


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    required_columns
].copy()

for column in required_columns:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


df = df[
    (df["Career"] != "") &
    (df["Job_Description"] != "")
].copy()


print(
    "Usable training rows:",
    len(df)
)


# ============================================================
# BUILD PROFILE + JOB DESCRIPTION PAIRS
# ============================================================

print("\nCreating SBERT training pairs...")

train_examples = []


for _, row in df.iterrows():

    profile_text = (
        f"Skills: {row['Skills']}. "
        f"Education: {row['Education']}. "
        f"Experience: {row['Experience']}."
    )

    job_text = (
        f"Job Description: "
        f"{row['Job_Description']}"
    )

    train_examples.append(
        InputExample(
            texts=[
                profile_text,
                job_text
            ]
        )
    )


print(
    "Training examples:",
    len(train_examples)
)


# ============================================================
# LOAD BASE SBERT
# ============================================================

print("\nLoading base Sentence-BERT model...")

model = SentenceTransformer(
    BASE_MODEL
)

print(
    "Base model:",
    BASE_MODEL
)


# ============================================================
# DATA LOADER
# ============================================================

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE
)


# ============================================================
# LOSS
# ============================================================

train_loss = (
    losses.MultipleNegativesRankingLoss(
        model
    )
)


# ============================================================
# TRAINING INFORMATION
# ============================================================

steps_per_epoch = max(
    1,
    len(train_dataloader)
)

warmup_steps = int(
    steps_per_epoch
    * EPOCHS
    * WARMUP_RATIO
)


print("\nTraining configuration:")
print(
    "Batch size       :",
    BATCH_SIZE
)

print(
    "Epochs           :",
    EPOCHS
)

print(
    "Steps per epoch  :",
    steps_per_epoch
)

print(
    "Warmup steps     :",
    warmup_steps
)


# ============================================================
# FINE-TUNING
# ============================================================

print("\n" + "=" * 75)
print("STARTING SBERT FINE-TUNING")
print("=" * 75)

model.fit(

    train_objectives=[
        (
            train_dataloader,
            train_loss
        )
    ],

    epochs=EPOCHS,

    warmup_steps=warmup_steps,

    show_progress_bar=True
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 75)
print("SAVING FINE-TUNED MODEL")
print("=" * 75)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

model.save(
    OUTPUT_DIR
)


print(
    "\nFine-tuned SBERT saved to:"
)

print(
    OUTPUT_DIR
)


print("\n" + "=" * 75)
print("TASK 2 SBERT FINE-TUNING COMPLETE")
print("=" * 75)
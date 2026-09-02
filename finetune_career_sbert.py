import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import torch
except ImportError:
    torch = None

try:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.evaluation import BinaryClassificationEvaluator
    
except ImportError as e:
    print("ERROR: Required package missing: " + str(e))
    print("Install with: pip install -U sentence-transformers==6.0.0 datasets")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG - easy to change
# ----------------------------------------------------------------------
RANDOM_STATE = 42
BASE_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 64
NUM_EPOCHS = 2
MAX_PAIRS_PER_CAREER = 150   # caps training set size; raise for more data, lower for speed
VAL_FRACTION_OF_TRAIN_POOL = 0.10  # internal split of the saved train_idx pool

DATASET_PATH = os.path.join("results", "milestone2_training", "career_profile_training_dataset.csv")
SPLIT_PATH = os.path.join("results", "milestone2_sentence_bert", "train_test_split_indices.npz")

OUTPUT_DIR = os.path.join("results", "milestone2_sentence_bert_finetuned")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "model")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

SCRIPT_START = time.time()


def log(msg):
    elapsed = time.time() - SCRIPT_START
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] (+{elapsed:.1f}s) {msg}")


def fail(msg, exc=None):
    print()
    print("=" * 70)
    print("ERROR: " + msg)
    if exc is not None:
        print("-" * 70)
        traceback.print_exc()
    print("=" * 70)
    sys.exit(1)


def build_positive_pairs(indices, skills_text, careers, max_pairs_per_career, rng):
    """Build non-overlapping (anchor, positive) pairs from same-career rows.
    Each row is used in at most one pair, so pairs never overlap."""
    by_career = defaultdict(list)
    for idx in indices:
        by_career[careers[idx]].append(idx)

    pairs = []
    skipped_careers = []
    for career, idx_list in by_career.items():
        if len(idx_list) < 2:
            skipped_careers.append(career)
            continue
        shuffled = idx_list[:]
        rng.shuffle(shuffled)
        n_pairs = min(max_pairs_per_career, len(shuffled) // 2)
        for i in range(n_pairs):
            a = shuffled[2 * i]
            b = shuffled[2 * i + 1]
            pairs.append((skills_text[a], skills_text[b]))
    return pairs, skipped_careers


def build_binary_eval_pairs(indices, skills_text, careers, n_positive, n_negative, rng):
    """Build labeled (sentence1, sentence2, label) pairs for evaluation only.
    label=1 for same-career pairs, label=0 for different-career pairs."""
    by_career = defaultdict(list)
    for idx in indices:
        by_career[careers[idx]].append(idx)

    sentence1, sentence2, labels = [], [], []

    # Positive pairs (non-overlapping within this call)
    pos_count = 0
    for career, idx_list in by_career.items():
        if len(idx_list) < 2 or pos_count >= n_positive:
            continue
        shuffled = idx_list[:]
        rng.shuffle(shuffled)
        a, b = shuffled[0], shuffled[1]
        sentence1.append(skills_text[a])
        sentence2.append(skills_text[b])
        labels.append(1)
        pos_count += 1

    # Negative pairs: sample two different careers at random
    careers_with_data = list(by_career.keys())
    neg_count = 0
    attempts = 0
    while neg_count < n_negative and attempts < n_negative * 20:
        attempts += 1
        c1, c2 = rng.sample(careers_with_data, 2)
        a = rng.choice(by_career[c1])
        b = rng.choice(by_career[c2])
        sentence1.append(skills_text[a])
        sentence2.append(skills_text[b])
        labels.append(0)
        neg_count += 1

    return sentence1, sentence2, labels


def main():
    print("=" * 70)
    print("CAREERCAST MILESTONE 2 - SBERT FINE-TUNING (career/skills proxy corpus)")
    print("=" * 70)
    print()
    print("NOTE: The available dataset has no real job-description sentence-pair")
    print("data. This script fine-tunes using label-based positive pairs (two")
    print("skills texts from the SAME career = a positive pair), which is a")
    print("practical proxy signal, not a genuine job-description similarity")
    print("corpus. This limitation is recorded in the saved training config.")
    print()

    rng = random.Random(RANDOM_STATE)

    # ------------------------------------------------------------------
    # STAGE 1: LOAD DATASET AND EXISTING SPLIT (reuse saved train_idx only)
    # ------------------------------------------------------------------
    log("Loading dataset and existing saved split...")
    try:
        if not os.path.isfile(DATASET_PATH):
            fail(f"Dataset not found at '{DATASET_PATH}'.")
        if not os.path.isfile(SPLIT_PATH):
            fail(f"Saved split file not found at '{SPLIT_PATH}'. "
                 f"Run the SBERT embedding script first.")

        df = pd.read_csv(DATASET_PATH)
        required_cols = {"skills", "career"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            fail(f"Dataset is missing required columns: {sorted(missing_cols)}")

        df = df.dropna(subset=["skills", "career"]).reset_index(drop=True)
        if len(df) == 0:
            fail("Dataset has 0 usable rows after dropping missing values.")

        split_data = np.load(SPLIT_PATH)
        saved_train_idx = split_data["train_idx"]

        if saved_train_idx.max() >= len(df):
            fail("Saved split indices are out of range for the current dataset.")
    except SystemExit:
        raise
    except Exception as e:
        fail(f"Failed to load dataset or split file: {e}", e)

    skills_text = df["skills"].astype(str).values
    careers = df["career"].astype(str).values

    log(f"Loaded {len(df)} rows. Using only the saved train_idx pool "
        f"({len(saved_train_idx)} rows) for fine-tuning; the original test_idx "
        f"is never touched.")

    # ------------------------------------------------------------------
    # STAGE 2: INTERNAL TRAIN/VAL SPLIT OF THE TRAIN POOL (no leakage)
    # ------------------------------------------------------------------
    log("Creating internal fine-tune train/validation split of the train pool...")
    try:
        pool_careers = careers[saved_train_idx]
        ft_train_idx, ft_val_idx = train_test_split(
            saved_train_idx,
            test_size=VAL_FRACTION_OF_TRAIN_POOL,
            random_state=RANDOM_STATE,
            stratify=pool_careers,
        )
    except Exception as e:
        fail(f"Internal train/validation split failed: {e}", e)

    overlap = set(ft_train_idx) & set(ft_val_idx)
    if overlap:
        fail(f"Data leakage detected: {len(overlap)} rows appear in both "
             f"fine-tune train and validation sets.")

    log(f"Internal split: {len(ft_train_idx)} fine-tune-train rows, "
        f"{len(ft_val_idx)} fine-tune-val rows. No overlap confirmed.")

    # ------------------------------------------------------------------
    # STAGE 3: BUILD TRAINING PAIRS AND EVALUATION PAIRS
    # ------------------------------------------------------------------
    log("Building positive training pairs from fine-tune-train rows...")
    try:
        train_pairs, skipped_train_careers = build_positive_pairs(
            ft_train_idx, skills_text, careers, MAX_PAIRS_PER_CAREER, rng
        )
        if len(train_pairs) == 0:
            fail("No training pairs could be constructed. Each career needs "
                 "at least 2 samples in the training pool.")
    except Exception as e:
        fail(f"Failed to build training pairs: {e}", e)

    log(f"Built {len(train_pairs)} training pairs "
        f"({len(skipped_train_careers)} careers skipped for insufficient samples).")

    log("Building labeled evaluation pairs from fine-tune-val rows...")
    try:
        n_eval_pairs = min(1000, len(ft_val_idx))
        eval_s1, eval_s2, eval_labels = build_binary_eval_pairs(
            ft_val_idx, skills_text, careers,
            n_positive=n_eval_pairs // 2, n_negative=n_eval_pairs // 2, rng=rng,
        )
        if len(eval_s1) == 0:
            fail("No evaluation pairs could be constructed from the validation pool.")
    except Exception as e:
        fail(f"Failed to build evaluation pairs: {e}", e)

    log(f"Built {len(eval_s1)} evaluation pairs "
        f"({sum(eval_labels)} positive, {len(eval_labels) - sum(eval_labels)} negative).")

    train_dataset = Dataset.from_dict({
        "anchor": [p[0] for p in train_pairs],
        "positive": [p[1] for p in train_pairs],
    })

    # ------------------------------------------------------------------
    # STAGE 4: LOAD BASE MODEL
    # ------------------------------------------------------------------
    log(f"Loading base model '{BASE_MODEL_NAME}'...")
    try:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        model = SentenceTransformer(BASE_MODEL_NAME, device=device)
    except Exception as e:
        fail(f"Failed to load base model '{BASE_MODEL_NAME}': {e}", e)
    log(f"Base model loaded on device '{device}'.")

    # ------------------------------------------------------------------
    # STAGE 4b: DIAGNOSTIC - similarity BEFORE fine-tuning (for later comparison)
    # ------------------------------------------------------------------
    log("Recording pre-fine-tuning similarity on a sample of evaluation pairs...")
    try:
        sample_n = min(10, len(eval_s1))
        pre_emb1 = model.encode(eval_s1[:sample_n], convert_to_numpy=True, normalize_embeddings=True)
        pre_emb2 = model.encode(eval_s2[:sample_n], convert_to_numpy=True, normalize_embeddings=True)
        pre_similarities = [float(np.dot(pre_emb1[i], pre_emb2[i])) for i in range(sample_n)]
    except Exception as e:
        fail(f"Pre-fine-tuning diagnostic encoding failed: {e}", e)

    # ------------------------------------------------------------------
    # STAGE 5: SET UP LOSS, EVALUATOR, TRAINER
    # ------------------------------------------------------------------
    log("Setting up loss function and evaluator...")
    try:
        loss = MultipleNegativesRankingLoss(model)

        evaluator = BinaryClassificationEvaluator(
            sentences1=eval_s1,
            sentences2=eval_s2,
            labels=eval_labels,
            name="career-pair-eval",
        )

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=CHECKPOINT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_ratio=0.1,
            fp16=(device == "cuda"),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=50,
            seed=RANDOM_STATE,
            report_to=[],
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=evaluator,
        )
    except Exception as e:
        fail(f"Failed to set up training components: {e}", e)
    log("Trainer set up successfully.")

    # ------------------------------------------------------------------
    # STAGE 6: TRAIN
    # ------------------------------------------------------------------
    log(f"Starting fine-tuning: {NUM_EPOCHS} epoch(s), batch size {BATCH_SIZE}, "
        f"{len(train_pairs)} training pairs, device '{device}'...")
    try:
        train_start = time.time()
        trainer.train()
        training_time = time.time() - train_start
    except Exception as e:
        fail(f"Fine-tuning failed: {e}", e)
    log(f"Fine-tuning complete in {training_time:.1f} seconds.")

    try:
        loss_history = trainer.state.log_history
    except Exception:
        loss_history = []

    # ------------------------------------------------------------------
    # STAGE 6b: DIAGNOSTIC - similarity AFTER fine-tuning (same sample pairs)
    # ------------------------------------------------------------------
    log("Recording post-fine-tuning similarity on the same sample pairs...")
    try:
        post_emb1 = model.encode(eval_s1[:sample_n], convert_to_numpy=True, normalize_embeddings=True)
        post_emb2 = model.encode(eval_s2[:sample_n], convert_to_numpy=True, normalize_embeddings=True)
        post_similarities = [float(np.dot(post_emb1[i], post_emb2[i])) for i in range(sample_n)]
    except Exception as e:
        fail(f"Post-fine-tuning diagnostic encoding failed: {e}", e)

    print()
    print("Sample pair similarity, pretrained vs fine-tuned (same pairs, same order):")
    print(f"{'label':>6} {'pretrained':>12} {'fine-tuned':>12}")
    for i in range(sample_n):
        print(f"{eval_labels[i]:>6} {pre_similarities[i]:>12.4f} {post_similarities[i]:>12.4f}")

    # ------------------------------------------------------------------
    # STAGE 7: EVALUATE FINAL MODEL ON THE FULL EVAL SET
    # ------------------------------------------------------------------
    log("Running final evaluator pass on the full evaluation pair set...")
    try:
        final_eval_result = evaluator(model)
    except Exception as e:
        fail(f"Final evaluation failed: {e}", e)
    log(f"Final evaluation result: {final_eval_result}")

    # ------------------------------------------------------------------
    # STAGE 8: SAVE MODEL AND ARTIFACTS
    # ------------------------------------------------------------------
    log("Saving fine-tuned model and artifacts...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save(MODEL_SAVE_DIR)

        config_path = os.path.join(OUTPUT_DIR, "training_config.json")
        stats_path = os.path.join(OUTPUT_DIR, "training_validation_stats.json")
        loss_history_path = os.path.join(OUTPUT_DIR, "loss_history.json")
        model_info_path = os.path.join(OUTPUT_DIR, "model_info.json")

        total_time = time.time() - SCRIPT_START

        training_config = {
            "base_model": BASE_MODEL_NAME,
            "fine_tuned": True,
            "fine_tuning_method": (
                "Label-based positive pairs: two 'skills' texts from the same "
                "'career' class treated as a positive pair; MultipleNegativesRankingLoss "
                "with in-batch negatives. This is a proxy for job-description semantic "
                "similarity constructed from the existing labeled dataset, NOT a genuine "
                "job-description corpus and NOT validated against SemEval or LinkedIn "
                "transition benchmarks."
            ),
            "loss_function": "MultipleNegativesRankingLoss",
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_pairs_per_career": MAX_PAIRS_PER_CAREER,
            "random_state": RANDOM_STATE,
            "device": device,
            "n_training_pairs": len(train_pairs),
            "n_evaluation_pairs": len(eval_s1),
            "careers_skipped_insufficient_samples": len(skipped_train_careers),
            "source_dataset": DATASET_PATH,
            "source_split_file": SPLIT_PATH,
            "leakage_control": (
                "Fine-tuning uses only rows from the saved train_idx pool; the "
                "original test_idx is never used. Fine-tune train/validation is "
                "an internal stratified split of that pool with no row overlap."
            ),
        }
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)

        training_validation_stats = {
            "n_fine_tune_train_rows": int(len(ft_train_idx)),
            "n_fine_tune_val_rows": int(len(ft_val_idx)),
            "n_training_pairs": len(train_pairs),
            "n_evaluation_pairs": len(eval_s1),
            "training_time_seconds": float(training_time),
            "total_time_seconds": float(total_time),
            "final_evaluator_result": (
                final_eval_result if isinstance(final_eval_result, (int, float))
                else str(final_eval_result)
            ),
            "sample_pair_similarity_pretrained": pre_similarities,
            "sample_pair_similarity_finetuned": post_similarities,
            "sample_pair_labels": eval_labels[:sample_n],
        }
        with open(stats_path, "w") as f:
            json.dump(training_validation_stats, f, indent=2)

        with open(loss_history_path, "w") as f:
            json.dump(loss_history, f, indent=2, default=str)

        model_info = {
            "model_name": BASE_MODEL_NAME + " (fine-tuned)",
            "is_fine_tuned": True,
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "saved_path": MODEL_SAVE_DIR,
            "device_used_for_training": device,
        }
        with open(model_info_path, "w") as f:
            json.dump(model_info, f, indent=2)
    except Exception as e:
        fail(f"Failed while saving artifacts: {e}", e)

    # ------------------------------------------------------------------
    # STAGE 9: VERIFY FILES
    # ------------------------------------------------------------------
    log("Verifying saved files...")
    expected_files = [config_path, stats_path, loss_history_path, model_info_path]
    missing_files = [p for p in expected_files if not os.path.isfile(p)]
    if not os.path.isdir(MODEL_SAVE_DIR) or not os.listdir(MODEL_SAVE_DIR):
        missing_files.append(MODEL_SAVE_DIR)
    if missing_files:
        fail(f"The following expected output files/directories were not found: {missing_files}")
    log("All output files verified.")

    total_time = time.time() - SCRIPT_START

    print()
    print("=" * 70)
    print("SBERT FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Fine-tuned model saved to: {MODEL_SAVE_DIR}")
    print(f"Training pairs used: {len(train_pairs)}")
    print(f"Evaluation pairs used: {len(eval_s1)}")
    print(f"Final evaluator result: {final_eval_result}")
    print(f"Training time: {training_time:.1f} seconds")
    print(f"Total time: {total_time:.1f} seconds")
    print()
    print("Saved files:")
    for p in expected_files:
        print(f"  {p}")
    print(f"  {MODEL_SAVE_DIR}/ (fine-tuned model)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail("Unhandled error in main().", e)
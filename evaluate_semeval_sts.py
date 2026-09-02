import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("Install it with: pip install -U sentence-transformers==6.0.0")
    sys.exit(1)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Extract both STS2017 zips into results/milestone2_semeval/ so you end up with:
#   results/milestone2_semeval/STS2017.eval.v1.1/STS.input.track5.en-en.txt
#   results/milestone2_semeval/STS2017.gs/STS.gs.track5.en-en.txt
SEMEVAL_DIR = os.path.join("results", "milestone2_semeval")
INPUT_PATH = os.path.join(SEMEVAL_DIR, "STS2017.eval.v1.1", "STS.input.track5.en-en.txt")
GOLD_PATH = os.path.join(SEMEVAL_DIR, "STS2017.gs", "STS.gs.track5.en-en.txt")

PRETRAINED_MODEL_NAME = "all-MiniLM-L6-v2"
FINETUNED_MODEL_DIR = os.path.join("results", "milestone2_sentence_bert_finetuned", "model")

OUTPUT_DIR = os.path.join("results", "milestone2_semeval")
BATCH_SIZE = 64

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


def load_sts_track(input_path, gold_path):
    if not os.path.isfile(input_path):
        fail(f"STS input file not found at '{input_path}'. "
             f"Extract STS2017.eval.v1.1 into '{SEMEVAL_DIR}'.")
    if not os.path.isfile(gold_path):
        fail(f"STS gold-standard file not found at '{gold_path}'. "
             f"Extract STS2017.gs into '{SEMEVAL_DIR}'.")

    sentence1, sentence2 = [], []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                fail(f"Malformed input line {line_num} in '{input_path}': "
                     f"expected 2 tab-separated fields, got {len(parts)}.")
            sentence1.append(parts[0])
            sentence2.append(parts[1])

    gold_scores = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                gold_scores.append(float(line))
            except ValueError:
                fail(f"Malformed gold-score line {line_num} in '{gold_path}': '{line}'")

    if len(sentence1) != len(gold_scores):
        fail(
            f"Line count mismatch: '{input_path}' has {len(sentence1)} pairs but "
            f"'{gold_path}' has {len(gold_scores)} gold scores. They must match "
            f"exactly to evaluate correctly."
        )

    return sentence1, sentence2, gold_scores


def evaluate_model(model_name, model, sentence1, sentence2, gold_scores):
    log(f"[{model_name}] Encoding sentence1 ({len(sentence1)} sentences)...")
    try:
        emb1 = model.encode(
            sentence1, batch_size=BATCH_SIZE, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        log(f"[{model_name}] Encoding sentence2 ({len(sentence2)} sentences)...")
        emb2 = model.encode(
            sentence2, batch_size=BATCH_SIZE, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
    except Exception as e:
        fail(f"[{model_name}] Encoding failed: {e}", e)

    # Embeddings are L2-normalized, so cosine similarity is a plain dot product.
    predicted_similarity = np.sum(emb1 * emb2, axis=1)

    # Gold scores are on a 0-5 scale; correlation is scale-invariant so no
    # rescaling is needed for Pearson/Spearman.
    pearson_corr, pearson_p = pearsonr(predicted_similarity, gold_scores)
    spearman_corr, spearman_p = spearmanr(predicted_similarity, gold_scores)

    log(f"[{model_name}] Pearson r = {pearson_corr:.4f} (p={pearson_p:.2e}), "
        f"Spearman rho = {spearman_corr:.4f} (p={spearman_p:.2e})")

    return {
        "model": model_name,
        "pearson_r": float(pearson_corr),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_corr),
        "spearman_p_value": float(spearman_p),
        "n_pairs": len(gold_scores),
        "predicted_similarity": predicted_similarity,
    }


def main():
    print("=" * 70)
    print("CAREERCAST MILESTONE 2 - SEMEVAL-2017 STS EVALUATION (track5 en-en)")
    print("=" * 70)
    print()
    print("NOTE: This benchmark uses generic SNLI-style English sentences, NOT")
    print("career/job-domain text. It validates general sentence-similarity")
    print("quality of the SBERT model, not career-specific matching quality.")
    print()

    # ------------------------------------------------------------------
    # STAGE 1: LOAD STS TRACK 5 DATA
    # ------------------------------------------------------------------
    log("Loading SemEval-2017 STS track5 (en-en) input and gold scores...")
    sentence1, sentence2, gold_scores = load_sts_track(INPUT_PATH, GOLD_PATH)
    log(f"Loaded {len(gold_scores)} sentence pairs with gold scores.")

    results = []

    # ------------------------------------------------------------------
    # STAGE 2: EVALUATE PRETRAINED MODEL
    # ------------------------------------------------------------------
    log(f"Loading pretrained model '{PRETRAINED_MODEL_NAME}'...")
    try:
        pretrained_model = SentenceTransformer(PRETRAINED_MODEL_NAME)
    except Exception as e:
        fail(f"Failed to load pretrained model '{PRETRAINED_MODEL_NAME}': {e}", e)
    log("Pretrained model loaded.")

    pretrained_result = evaluate_model(
        "pretrained_all-MiniLM-L6-v2", pretrained_model, sentence1, sentence2, gold_scores
    )
    results.append(pretrained_result)

    # ------------------------------------------------------------------
    # STAGE 3: EVALUATE FINE-TUNED MODEL (if available)
    # ------------------------------------------------------------------
    finetuned_result = None
    if os.path.isdir(FINETUNED_MODEL_DIR) and os.listdir(FINETUNED_MODEL_DIR):
        log(f"Loading fine-tuned model from '{FINETUNED_MODEL_DIR}'...")
        try:
            finetuned_model = SentenceTransformer(FINETUNED_MODEL_DIR)
        except Exception as e:
            fail(f"Failed to load fine-tuned model from '{FINETUNED_MODEL_DIR}': {e}", e)
        log("Fine-tuned model loaded.")

        finetuned_result = evaluate_model(
            "finetuned_career_sbert", finetuned_model, sentence1, sentence2, gold_scores
        )
        results.append(finetuned_result)
    else:
        log(f"Fine-tuned model not found at '{FINETUNED_MODEL_DIR}' - "
            f"skipping fine-tuned evaluation. Run finetune_career_sbert.py first "
            f"if you want this comparison.")

    # ------------------------------------------------------------------
    # STAGE 4: SAVE ARTIFACTS
    # ------------------------------------------------------------------
    log("Saving results...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        predictions_path = os.path.join(OUTPUT_DIR, "semeval_sts_track5_predictions.csv")
        pred_data = {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "gold_score": gold_scores,
        }
        for r in results:
            col_name = f"predicted_similarity_{r['model']}"
            pred_data[col_name] = r["predicted_similarity"]
        pd.DataFrame(pred_data).to_csv(predictions_path, index=False)

        results_path = os.path.join(OUTPUT_DIR, "semeval_sts_results.json")
        total_time = time.time() - SCRIPT_START
        summary = {
            "benchmark": "SemEval-2017 STS Shared Task, track5 (en-en)",
            "benchmark_domain_note": (
                "Generic SNLI-derived English sentences, not career/job-domain "
                "text. Measures general sentence-similarity quality only."
            ),
            "n_pairs": len(gold_scores),
            "results": [
                {
                    "model": r["model"],
                    "pearson_r": r["pearson_r"],
                    "pearson_p_value": r["pearson_p_value"],
                    "spearman_rho": r["spearman_rho"],
                    "spearman_p_value": r["spearman_p_value"],
                }
                for r in results
            ],
            "total_time_seconds": float(total_time),
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        fail(f"Failed while saving results: {e}", e)

    # ------------------------------------------------------------------
    # STAGE 5: VERIFY FILES
    # ------------------------------------------------------------------
    log("Verifying saved files...")
    expected_files = [predictions_path, results_path]
    missing_files = [p for p in expected_files if not os.path.isfile(p)]
    if missing_files:
        fail(f"The following expected output files were not found: {missing_files}")
    log("All output files verified.")

    total_time = time.time() - SCRIPT_START

    print()
    print("=" * 70)
    print("SEMEVAL-2017 STS EVALUATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Track: track5 (en-en), {len(gold_scores)} pairs")
    print()
    for r in results:
        print(f"{r['model']}:")
        print(f"  Pearson r  = {r['pearson_r']:.4f}")
        print(f"  Spearman rho = {r['spearman_rho']:.4f}")
        print()
    if finetuned_result is None:
        print("(Fine-tuned model was not evaluated - not found on disk.)")
        print()
    print(f"Total time: {total_time:.1f} seconds")
    print()
    print("Saved files:")
    for p in expected_files:
        print(f"  {p}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail("Unhandled error in main().", e)
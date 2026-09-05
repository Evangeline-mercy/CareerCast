"""Create a small, reproducible t-SNE projection for the CareerCast dashboard."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


EMBEDDINGS_PATH = Path(
    "results/milestone2_sentence_bert_finetuned/"
    "sentence_bert_embeddings_finetuned.npy"
)
DATASET_PATH = Path(
    "results/milestone2_training/career_profile_training_dataset.csv"
)
OUTPUT_PATH = Path("streamlit_app/assets/finetuned_sbert_tsne.csv")
RANDOM_STATE = 42
SAMPLES_PER_CAREER = 10


def main() -> None:
    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Missing embeddings: {EMBEDDINGS_PATH}")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Missing training dataset: {DATASET_PATH}")

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    dataset = pd.read_csv(DATASET_PATH)
    if "career" not in dataset.columns:
        raise ValueError("Training dataset must contain a 'career' column.")
    if len(dataset) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: dataset={len(dataset)}, embeddings={embeddings.shape[0]}"
        )

    sampled = (
        dataset.reset_index(names="row_index")
        .groupby("career", group_keys=False, sort=True)
        .sample(n=SAMPLES_PER_CAREER, random_state=RANDOM_STATE)
        .sort_values(["career", "row_index"])
        .reset_index(drop=True)
    )
    row_indices = sampled["row_index"].to_numpy(dtype=int)
    sampled_embeddings = np.asarray(embeddings[row_indices], dtype=np.float32)

    pca_components = min(50, sampled_embeddings.shape[0] - 1, sampled_embeddings.shape[1])
    reduced = PCA(n_components=pca_components, random_state=RANDOM_STATE).fit_transform(
        sampled_embeddings
    )
    projection = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=RANDOM_STATE,
    ).fit_transform(reduced)

    output = pd.DataFrame(
        {
            "row_index": row_indices,
            "career": sampled["career"].astype(str),
            "tsne_x": projection[:, 0],
            "tsne_y": projection[:, 1],
        }
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")
    print(f"Careers: {output['career'].nunique()}")
    print("Existing models were not modified.")


if __name__ == "__main__":
    main()

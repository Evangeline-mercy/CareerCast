# Model card: CareerCast career classifiers

## Model overview

CareerCast embeds skill-profile text with a career-domain fine-tuned
`all-MiniLM-L6-v2` Sentence-BERT model and compares three supervised
classifiers across 96 career classes. The embedding dimension is 384. The
selected single prediction model is Logistic Regression; the recommendation
endpoint uses a weighted three-model ensemble. Runtime selection is atomic: it
does not combine a fine-tuned encoder with legacy classifiers.

## Training and selection

- Dataset rows: 48,000
- Training samples: 38,400
- Held-out testing samples: 9,600
- Random state: 42
- Cross-validation folds: 2
- Balanced CV tuning subset: 3,840 training-only samples (40 per class)
- Final estimator refit: all 38,400 training samples
- Selection metric: cross-validation accuracy
- Test set used for model selection: no

## Recorded evaluation

| Classifier | CV accuracy | Test accuracy | Macro precision | Macro recall | Macro F1 | Parameters |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.9986979 | 0.9996875 | 0.9996885 | 0.9996875 | 0.9996875 | `C=10.0` |
| Random Forest | 0.9986979 | 0.9995833 | 0.9995854 | 0.9995833 | 0.9995833 | `n_estimators=100`, `max_depth=None`, `max_features=sqrt` |
| XGBoost | 0.9966146 | 0.9996875 | 0.9996885 | 0.9996875 | 0.9996875 | `n_estimators=100`, `max_depth=3` |

Metrics are read from
`results/milestone2_finetuned_sbert_classifiers/finetuned_classifier_summary.json`.

On the same saved split, fine-tuned SBERT plus Logistic Regression improved
test accuracy from `0.9982292` to `0.9996875` and macro F1 from `0.9982233`
to `0.9996875`. SemEval-2017 STS track 5 evaluation produced Pearson
`0.8808883` and Spearman `0.8793690` for the fine-tuned encoder. This benchmark
contains general English sentence pairs and does not establish career-domain
prediction accuracy.

## Runtime behavior

- `/predict` uses Logistic Regression probabilities.
- `/recommend` defaults to LR `0.40`, RF `0.30`, and XGBoost `0.30`.
- `/gap-report` compares normalized candidate skills with weighted career
  requirements and returns priorities and learning actions.
- `/models/info` discloses the active pipeline and whether its encoder is
  fine-tuned.

## Intended use

- Educational career exploration
- Comparing possible career paths from supplied skills
- Identifying potential learning priorities
- Demonstrating an end-to-end ML API and review interface

## Limitations and risks

- Near-perfect prepared-dataset results may reflect strongly separable or
  generated profiles and must not be interpreted as real-world accuracy.
- Held-out accuracy is an aggregate evaluation metric; it is not the same as
  the confidence score for an individual profile.
- Probabilities are model scores, not guarantees of aptitude, employability, or
  success.
- Skills alone cannot represent motivation, experience, opportunity, location,
  accessibility, or personal preference.
- The model has not been documented as independently audited for demographic
  fairness, calibration, distribution shift, or adversarial inputs.
- Career and skill definitions can become outdated and require periodic review.

## Responsible-use guidance

Use CareerCast as one input to a broader discussion with the learner. Show
alternative careers, disclose uncertainty, allow users to correct extracted
skills, and never use the output as the sole basis for a high-impact decision.

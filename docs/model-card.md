# Model card: CareerCast career classifiers

## Model overview

CareerCast embeds skill-profile text with pretrained `all-MiniLM-L6-v2`
Sentence-BERT embeddings and compares three supervised classifiers across 96
career classes. Sentence-BERT was not fine-tuned. The embedding dimension is
384. The selected single prediction model is Logistic Regression; the
recommendation endpoint uses a weighted three-model ensemble.

## Training and selection

- Dataset rows: 48,000
- Training samples: 38,400
- Held-out testing samples: 9,600
- Random state: 42
- Cross-validation folds: 2
- Selection metric: cross-validation accuracy
- Test set used for model selection: no

## Recorded evaluation

| Classifier | CV accuracy | Test accuracy | Macro precision | Macro recall | Macro F1 | Parameters |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.9978125 | 0.9982292 | 0.9984010 | 0.9982292 | 0.9982233 | `C=10.0` |
| Random Forest | 0.9961719 | 0.9972917 | 0.9974095 | 0.9972917 | 0.9972849 | `n_estimators=100`, `max_depth=None` |
| XGBoost | 0.9963281 | 0.9970833 | 0.9971153 | 0.9970833 | 0.9970784 | `n_estimators=100`, `max_depth=5` |

Metrics are read from
`results/milestone2_sentence_bert_classifier/sbert_classifier_summary.json`.

## Runtime behavior

- `/predict` uses Logistic Regression probabilities.
- `/recommend` defaults to LR `0.40`, RF `0.30`, and XGBoost `0.30`.
- `/gap-report` compares normalized candidate skills with weighted career
  requirements and returns priorities and learning actions.

## Intended use

- Educational career exploration
- Comparing possible career paths from supplied skills
- Identifying potential learning priorities
- Demonstrating an end-to-end ML API and review interface

## Limitations and risks

- Near-perfect prepared-dataset results may reflect strongly separable or
  generated profiles and must not be interpreted as real-world accuracy.
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

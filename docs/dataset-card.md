# Dataset card: CareerCast career profiles

## Summary

The packaged training dataset contains skill profiles paired with career labels
for supervised career-classification experiments.

| Property | Recorded value |
|---|---:|
| Rows | 48,000 |
| Career classes | 96 |
| Training samples | 38,400 |
| Testing samples | 9,600 |
| Random state | 42 |
| File | `results/milestone2_training/career_profile_training_dataset.csv` |

## Schema

| Column | Type | Description |
|---|---|---|
| `career` | string | Target career label |
| `skills` | string | Text representation of a candidate skill profile |

The model pipeline converts the `skills` text into a 384-dimensional embedding.

## Related skill-gap evidence

Weighted career requirements may be loaded from
`datasets/career_skill_weights.csv`. When curated weights are unavailable for a
career, CareerCast derives fallback weights from skill frequencies in the
training profiles. The runtime reported 105 available gap profiles during the
Milestone 4 verification.

## Intended use

- Developing and evaluating CareerCast career classifiers
- Demonstrating career ranking and skill-gap analysis
- Integration, regression, and accuracy-gate testing

## Limitations

- Career labels and required skills are simplified representations of changing
  labour-market roles.
- High scores on this prepared dataset do not establish equal performance on
  real resumes or across demographic, regional, educational, or language groups.
- The two-column schema does not capture interests, work history, constraints,
  qualifications, or career satisfaction.
- Dataset provenance and reuse rights must be checked against the repository's
  upstream source files before redistribution outside this project.

## Privacy and responsible use

The published training schema contains career and skills fields, not candidate
identity fields. Do not add names, contact details, or other personal data to a
public training release. CareerCast output must not be used as the sole basis
for employment, admission, or other high-impact decisions.

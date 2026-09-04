# Testing and quality gates

## Verified suites

CareerCast includes automated coverage for:

- Skill parsing, normalization, aliases, and priority boundaries
- FastAPI health, prediction, recommendation, and gap-report contracts
- End-to-end parsing-to-gap and automatic-target flows
- Ensemble ranking, weight normalization, and invalid inputs
- Python package client and CLI forwarding
- Cohort CSV validation, distribution, and summary calculations
- PDF content generation
- Reproducible accuracy threshold

## Local command

```bash
python -m pytest \
  tests/test_gap_analysis_ci.py \
  tests/test_api_contract.py \
  tests/test_pdf_report.py \
  tests/test_accuracy_gate.py \
  tests/test_package_parsing.py \
  tests/test_package_client.py \
  tests/test_cli.py \
  tests/test_pipeline_integration.py \
  tests/test_pipeline_regression.py \
  tests/test_cohort_analytics.py \
  tests/test_publication_docs.py \
  -v --assert=plain -s
```

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` in resource-constrained
environments.

## Continuous integration

`.github/workflows/ci.yml` installs the package, compiles critical modules,
runs all focused suites, builds both wheel and source distributions, and applies
the reproducible accuracy gate on pushes and pull requests to `master`.

The required accuracy threshold is `0.80`. The deterministic CI fixture records
`1.00`; this fixture result verifies the gate mechanism and is separate from the
production model metrics documented in the model card.

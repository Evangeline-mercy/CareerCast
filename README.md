# CareerCast

CareerCast is an AI-assisted career decision-support system that predicts likely
career paths, ranks ensemble recommendations, identifies weighted skill gaps,
and suggests practical learning actions.

[Open the deployed CareerCast application](https://careercast-milestone3.streamlit.app/)

## Features

- FastAPI prediction, recommendation, model-information, and skill-gap endpoints
- Sentence-BERT profile embeddings with three evaluated classifiers
- Weighted, explainable skill-gap analysis with priority levels and actions
- Pip-installable Python client and `careercast` command-line interface
- Individual review, cohort analytics, and side-by-side career comparison
- Downloadable PDF career reports and cohort CSV results
- Automated integration, regression, PDF, and accuracy-gate tests

## Installation

Clone the repository and install the package from the project root:

```bash
python -m pip install -e .
```

For development and package-building tools:

```bash
python -m pip install -e ".[dev]"
```

## Start locally

Start the API from the project root:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start Streamlit:

```bash
python -m streamlit run streamlit_app/app.py
```

Interactive FastAPI documentation is available locally at
`http://127.0.0.1:8000/docs`.

## Python client

```python
from careercast import CareerCastClient

client = CareerCastClient("http://127.0.0.1:8000")
result = client.predict("Python, SQL, pandas", top_k=5)
print(result["top_predictions"])
```

## Command-line interface

```bash
careercast health
careercast predict "Python, SQL, pandas" --top-k 5
careercast recommend "Python, SQL, machine learning" --top-k 5
careercast gap "Python, SQL" --target-career "Data Scientist"
```

Place the global `--api-url` option before the command when using another API:

```bash
careercast --api-url http://127.0.0.1:8000 health
```

## Documentation

- [Documentation index](docs/README.md)
- [REST API reference](docs/api-reference.md)
- [CLI reference](docs/cli-reference.md)
- [Dataset card](docs/dataset-card.md)
- [Model card](docs/model-card.md)
- [Testing and quality gates](docs/testing.md)
- [Deployment guide](docs/deployment.md)
- [Public release checklist](docs/release-checklist.md)

## Evaluation summary

The selected Logistic Regression classifier recorded `0.9978125` two-fold
cross-validation accuracy and `0.9982292` held-out test accuracy on the prepared
48,000-row career-profile dataset. See the model card for all model results and
important limitations.

## Responsible use

CareerCast provides decision-support scores, not guaranteed career outcomes.
Recommendations should be combined with personal interests, education,
experience, accessibility needs, and advice from qualified mentors.

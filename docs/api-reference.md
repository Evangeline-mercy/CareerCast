# REST API reference

The FastAPI application is defined in `api/main.py`. Start it from the repository
root with:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Base URL for local examples: `http://127.0.0.1:8000`.

## GET /health

Reports API status, loaded model flags, and profile counts.

```json
{
  "status": "ok",
  "models_loaded": {"sbert": true, "lr": true, "rf": true, "xgb": true, "le": true},
  "career_profiles_loaded": 96,
  "gap_profiles_loaded": 105
}
```

## GET /models/info

Returns embedding configuration, number of career classes, classifier names,
stored metrics, and available career-profile count.

## POST /predict

Returns Logistic Regression Top-K career predictions.

Request fields:

| Field | Type | Required | Rules |
|---|---|---:|---|
| `skills_text` | string | Yes | Must not be blank |
| `top_k` | integer | No | Default 5; between 1 and 20 |

```json
{"skills_text": "Python, SQL, pandas", "top_k": 5}
```

The response contains `top_predictions`, `embedding_model`, and a shortened copy
of `input_text`. Each prediction contains `rank`, `career`, `probability`, and
`model`.

## POST /recommend

Returns Top-K ensemble recommendations with per-model probabilities.

| Field | Type | Required | Rules |
|---|---|---:|---|
| `skills_text` | string | Yes | Must not be blank |
| `top_k` | integer | No | Default 10; between 1 and 20 |
| `ensemble_weights` | object | No | Non-negative `lr`, `rf`, and `xgb` values |

Default weights are LR `0.40`, RF `0.30`, and XGBoost `0.30`. Supplied weights
are normalized to sum to one and at least one must be positive.

```json
{
  "skills_text": "Python, SQL, machine learning",
  "top_k": 5,
  "ensemble_weights": {"lr": 0.4, "rf": 0.3, "xgb": 0.3}
}
```

## POST /gap-report

Generates weighted skill-gap evidence and actionable learning suggestions.

| Field | Type | Required | Rules |
|---|---|---:|---|
| `skills_text` | string | Yes | Comma, semicolon, pipe, or newline separators |
| `target_career` | string or null | No | Uses ensemble prediction when omitted |
| `top_k_careers` | integer | No | Default 5; between 1 and 10 |

```json
{
  "skills_text": "Python, SQL, pandas",
  "target_career": "Data Scientist",
  "top_k_careers": 1
}
```

The response includes normalized `candidate_skills`, the primary
`target_career`, one or more `gap_analysis` records, and `top_missing_skills`.
Each gap record includes matched skills, weighted missing skills, alignment
score, evidence source, priority summary, and suggestions.

## Errors

- `400`: blank input, invalid Top-K value, or invalid ensemble weights
- `404`: no skill profile exists for the requested target career
- `422`: request body does not match the schema
- `503`: a required model or skill-gap analyzer is unavailable

FastAPI also exposes live OpenAPI documentation at `/docs` and `/redoc`.

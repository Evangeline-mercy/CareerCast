# Deployment guide

## Public interface

CareerCast is deployed at:

https://careercast-milestone3.streamlit.app/

The Streamlit Community Cloud deployment runs `streamlit_app/app.py`. With
`CAREERCAST_EMBEDDED_API=1`, `deployment/bootstrap.py` downloads missing model
artifacts from the project's GitHub release and starts FastAPI inside the hosted
runtime.

## Required Streamlit secrets

```toml
CAREERCAST_EMBEDDED_API = "1"
OMP_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
```

These values are configuration rather than credentials.

## Model artifacts

Deployment requires Logistic Regression, Random Forest, XGBoost, label encoder,
classifier summary, and career-profile training artifacts. Large binaries are
kept outside ordinary Git history and fetched by the bootstrap process.

## Local deployment

Run FastAPI and Streamlit in separate terminals as described in the repository
README. The local API binds to `127.0.0.1:8000`; the Streamlit UI normally uses
`127.0.0.1:8501`.

## Verification

After deployment, confirm:

1. The sidebar displays `API connected` and non-zero profile counts.
2. Individual prediction, recommendations, gap analysis, and PDF download work.
3. Cohort template upload produces metrics, distribution, results, and CSV export.
4. Career Comparison accepts two different recommended careers.
5. The latest GitHub Actions workflow is green.

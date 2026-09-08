# CareerCast Supabase Authentication Integration

This package adds authentication without changing the trained SBERT model, Logistic Regression,
Random Forest, XGBoost, ensemble weights, career profiles, skill-gap calculations, dashboards, or
report generation.

## Files to copy

Copy the package contents into the CareerCast repository root and allow the existing files to be
replaced. The new files are `streamlit_app/auth.py`, `api/auth.py`, and
`tests/test_auth.py`.

## Local secrets

Keep `.streamlit/secrets.toml` untracked. Use the publishable key only:

```toml
[supabase]
url = "https://YOUR_PROJECT_REFERENCE.supabase.co"
publishable_key = "YOUR_SB_PUBLISHABLE_KEY"
redirect_url = "http://localhost:8501/"
```

Never add a Supabase secret key, service-role key, JWT signing secret, or database password to this
file for CareerCast.

## Test locally

Terminal 1:

```bat
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bat
python -m streamlit run streamlit_app\app.py
```

Open `http://localhost:8501`, create an account, confirm the message sent to the email address, and
then sign in. Verify Individual Review, Milestone 2 Analytics, Cohort Analytics, and Secure logout.

Run the focused tests:

```bat
python -m pytest tests\test_auth.py tests\test_api_contract.py tests\test_pipeline_integration.py tests\test_pipeline_regression.py -v --assert=plain
```

## Streamlit Cloud secrets

In the deployed app, open **Manage app > Settings > Secrets** and add:

```toml
[supabase]
url = "https://YOUR_PROJECT_REFERENCE.supabase.co"
publishable_key = "YOUR_SB_PUBLISHABLE_KEY"
redirect_url = "https://careercast-milestone3.streamlit.app/"
```

Save and reboot the app. Do not paste the credentials into GitHub.

## Rollback behavior

Authentication is deliberately configuration-gated. If the Supabase URL or publishable key is not
configured, the existing CareerCast application continues to operate and its existing CI tests stay
compatible. Once both settings are supplied, the UI login gate and API protection activate together.

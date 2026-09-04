# CareerCast

CareerCast provides AI-powered career prediction, ensemble recommendations,
and weighted skill-gap analysis through a REST API, Python client, CLI, and
Streamlit review interface.

## Install the Python package

```bash
python -m pip install -e .
```

## CLI examples

```bash
careercast health
careercast predict "Python, SQL, pandas" --top-k 5
careercast recommend "Python, SQL, machine learning" --top-k 5
careercast gap "Python, SQL" --target-career "Data Scientist"
```

The CLI uses `http://127.0.0.1:8000` by default. Override it with
`--api-url` or the `CAREERCAST_API_URL` environment variable.

## Python example

```python
from careercast import CareerCastClient

client = CareerCastClient("http://127.0.0.1:8000")
result = client.predict("Python, SQL, pandas", top_k=5)
print(result["top_predictions"])
```

Public review UI: https://careercast-milestone3.streamlit.app/

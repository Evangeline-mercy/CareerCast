"""Prepare release artifacts and run FastAPI beside Streamlit in deployment."""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from pathlib import Path


RELEASE_BASE_URL = (
    "https://github.com/Evangeline-mercy/CareerCast/releases/download/v1.0.0"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = PROJECT_ROOT / "results" / "milestone2_sentence_bert_classifier"
TRAINING_DIR = PROJECT_ROOT / "results" / "milestone2_training"

ARTIFACTS = {
    CLASSIFIER_DIR / "logistic_regression_model.pkl": "logistic_regression_model.pkl",
    CLASSIFIER_DIR / "random_forest_model.pkl": "random_forest_model.pkl",
    CLASSIFIER_DIR / "xgboost_model.pkl": "xgboost_model.pkl",
    CLASSIFIER_DIR / "label_encoder.pkl": "label_encoder.pkl",
    CLASSIFIER_DIR / "sbert_classifier_summary.json": "sbert_classifier_summary.json",
    TRAINING_DIR / "career_profile_training_dataset.csv": "career_profile_training_dataset.csv",
}

_server_thread: threading.Thread | None = None


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "CareerCast/3.0"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_artifacts() -> None:
    """Download only artifacts that are absent from the deployment filesystem."""
    for destination, filename in ARTIFACTS.items():
        if destination.exists() and destination.stat().st_size > 0:
            continue
        _download(f"{RELEASE_BASE_URL}/{filename}", destination)


def start_embedded_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the existing FastAPI application once in a background thread."""
    global _server_thread
    if _server_thread is not None and _server_thread.is_alive():
        return

    ensure_artifacts()
    os.environ["CAREERCAST_ALLOW_MODEL_DOWNLOAD"] = "1"

    def run() -> None:
        import uvicorn

        uvicorn.run("api.main:app", host=host, port=port, log_level="info")

    _server_thread = threading.Thread(target=run, name="careercast-api", daemon=True)
    _server_thread.start()

    health_url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + 300
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"CareerCast API did not start within 300 seconds: {last_error}")

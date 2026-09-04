"""Small, typed-friendly HTTP client for the CareerCast REST API."""

from __future__ import annotations

from typing import Any, Mapping

import requests


class CareerCastAPIError(RuntimeError):
    """Raised when the CareerCast service rejects or cannot serve a request."""


class CareerCastClient:
    """Access prediction, recommendation, and skill-gap endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = str(exc)
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    detail = response.json().get("detail", detail)
                except (ValueError, AttributeError):
                    pass
            raise CareerCastAPIError(detail) from exc
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def models_info(self) -> dict[str, Any]:
        return self._request("GET", "/models/info")

    def predict(self, skills_text: str, *, top_k: int = 5) -> dict[str, Any]:
        return self._request(
            "POST", "/predict", json={"skills_text": skills_text, "top_k": top_k}
        )

    def recommend(
        self,
        skills_text: str,
        *,
        top_k: int = 10,
        ensemble_weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"skills_text": skills_text, "top_k": top_k}
        if ensemble_weights is not None:
            payload["ensemble_weights"] = dict(ensemble_weights)
        return self._request("POST", "/recommend", json=payload)

    def gap_report(
        self,
        skills_text: str,
        *,
        target_career: str | None = None,
        top_k_careers: int = 5,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/gap-report",
            json={
                "skills_text": skills_text,
                "target_career": target_career,
                "top_k_careers": top_k_careers,
            },
        )

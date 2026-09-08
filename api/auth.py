"""Optional Supabase bearer-token validation for CareerCast API endpoints."""

from __future__ import annotations

import os
from pathlib import Path

import requests
from fastapi import Header, HTTPException


def auth_config() -> tuple[str, str]:
    """Read environment configuration, then the ignored local Streamlit secrets file."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    if url and key:
        return url.rstrip("/"), key
    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if secrets_path.is_file():
        try:
            import tomllib

            with secrets_path.open("rb") as handle:
                section = tomllib.load(handle).get("supabase", {})
            url = str(section.get("url", "")).strip()
            key = str(section.get("publishable_key", "")).strip()
        except (ImportError, OSError, ValueError, TypeError):
            return "", ""
    return url.rstrip("/"), key


def auth_enabled() -> bool:
    url, key = auth_config()
    return bool(url and key)


def require_authenticated_user(authorization: str | None = Header(default=None)) -> dict | None:
    """Validate a bearer token with Supabase; remain compatible when auth is not configured."""
    if not auth_enabled():
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    url, key = auth_config()
    try:
        response = requests.get(
            f"{url}/auth/v1/user",
            headers={"apikey": key, "Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return response.json()

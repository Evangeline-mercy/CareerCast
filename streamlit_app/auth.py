"""Supabase email/password authentication for the CareerCast Streamlit UI."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


def _secret(name: str, default: str = "") -> str:
    """Read a Supabase setting from Streamlit secrets without exposing it."""
    try:
        section = st.secrets.get("supabase", {})
        value = section.get(name, "") if hasattr(section, "get") else ""
        if value:
            return str(value).strip()
        aliases = {
            "url": ("SUPABASE_URL", "supabase_url"),
            "publishable_key": ("SUPABASE_PUBLISHABLE_KEY", "supabase_publishable_key"),
            "redirect_url": ("SUPABASE_REDIRECT_URL", "supabase_redirect_url"),
        }
        for alias in aliases.get(name, ()):
            value = st.secrets.get(alias, "")
            if value:
                return str(value).strip()
    except (FileNotFoundError, KeyError, TypeError):
        pass
    env_name = {
        "url": "SUPABASE_URL",
        "publishable_key": "SUPABASE_PUBLISHABLE_KEY",
        "redirect_url": "SUPABASE_REDIRECT_URL",
    }[name]
    return os.getenv(env_name, default).strip()


def auth_config() -> tuple[str, str, str]:
    return (
        _secret("url").rstrip("/"),
        _secret("publishable_key"),
        _secret("redirect_url"),
    )


def configure_auth_environment() -> bool:
    """Pass public auth configuration to the embedded FastAPI process."""
    url, key, _ = auth_config()
    if not (url and key):
        return False
    os.environ["SUPABASE_URL"] = url
    os.environ["SUPABASE_PUBLISHABLE_KEY"] = key
    return True


def access_token() -> str:
    return str(st.session_state.get("auth_access_token", ""))


def clear_session() -> None:
    for key in ("auth_access_token", "auth_refresh_token", "auth_user_email", "career_result"):
        st.session_state.pop(key, None)


def _auth_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    url, key, _ = auth_config()
    if not (url and key):
        raise RuntimeError("Supabase authentication is not configured.")
    headers = {"apikey": key, "Content-Type": "application/json"}
    headers.update(kwargs.pop("headers", {}))
    return requests.request(method, f"{url}/auth/v1/{path}", headers=headers, timeout=20, **kwargs)


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
        return str(body.get("msg") or body.get("message") or body.get("error_description") or body.get("error"))
    except (ValueError, AttributeError):
        return "Authentication request failed."


def sign_in(email: str, password: str) -> None:
    response = _auth_request(
        "POST", "token?grant_type=password", json={"email": email.strip(), "password": password}
    )
    if not response.ok:
        raise RuntimeError(_error_message(response))
    body = response.json()
    st.session_state["auth_access_token"] = body["access_token"]
    st.session_state["auth_refresh_token"] = body.get("refresh_token", "")
    st.session_state["auth_user_email"] = body.get("user", {}).get("email", email.strip())


def sign_up(email: str, password: str) -> bool:
    _, _, redirect_url = auth_config()
    payload: dict[str, Any] = {"email": email.strip(), "password": password}
    if redirect_url:
        payload["options"] = {"email_redirect_to": redirect_url}
    response = _auth_request("POST", "signup", json=payload)
    if not response.ok:
        raise RuntimeError(_error_message(response))
    body = response.json()
    if body.get("access_token"):
        st.session_state["auth_access_token"] = body["access_token"]
        st.session_state["auth_refresh_token"] = body.get("refresh_token", "")
        st.session_state["auth_user_email"] = body.get("user", {}).get("email", email.strip())
        return True
    return False


def sign_out() -> None:
    token = access_token()
    if token:
        try:
            _auth_request("POST", "logout", headers={"Authorization": f"Bearer {token}"})
        except requests.RequestException:
            pass
    clear_session()


def render_auth_gate() -> bool:
    """Render login/registration when configured; return whether app access is allowed."""
    url, key, _ = auth_config()
    if not (url and key):
        st.sidebar.warning("Authentication is not configured")
        return True  # Safe staged rollout: existing deployments remain usable until secrets are added.

    if access_token():
        st.sidebar.caption(f"Signed in as {st.session_state.get('auth_user_email', 'CareerCast user')}")
        if st.sidebar.button("Secure logout", width="stretch"):
            sign_out()
            st.rerun()
        return True

    st.title("CareerCast secure access")
    st.caption("Sign in to access career predictions, recommendations and cohort analytics.")
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("careercast_login"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            if not email.strip() or not password:
                st.warning("Enter your email and password.")
            else:
                try:
                    sign_in(email, password)
                    st.rerun()
                except (requests.RequestException, RuntimeError, KeyError) as exc:
                    st.error(f"Sign-in failed: {exc}")

    with register_tab:
        with st.form("careercast_register"):
            new_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm = st.text_input("Confirm password", type="password", key="register_confirm")
            registered = st.form_submit_button("Create account", width="stretch")
        if registered:
            if not new_email.strip() or not new_password:
                st.warning("Enter an email and password.")
            elif len(new_password) < 8:
                st.warning("Use a password with at least 8 characters.")
            elif new_password != confirm:
                st.warning("Passwords do not match.")
            else:
                try:
                    active = sign_up(new_email, new_password)
                    if active:
                        st.rerun()
                    st.success("Account created. Check your email and confirm it before signing in.")
                except (requests.RequestException, RuntimeError, KeyError) as exc:
                    st.error(f"Registration failed: {exc}")
    return False

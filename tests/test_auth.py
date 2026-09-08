"""Authentication tests that do not contact Supabase."""

from fastapi import HTTPException

import api.auth as auth


def test_auth_is_optional_without_configuration(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    assert auth.require_authenticated_user(None) is None


def test_missing_bearer_token_is_rejected_when_enabled(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "test-publishable-key")
    try:
        auth.require_authenticated_user(None)
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Expected missing token to be rejected")


def test_valid_token_returns_user(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "test-publishable-key")

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"id": "user-1", "email": "user@example.com"}

    monkeypatch.setattr(auth.requests, "get", lambda *args, **kwargs: Response())
    user = auth.require_authenticated_user("Bearer valid-token")
    assert user["id"] == "user-1"

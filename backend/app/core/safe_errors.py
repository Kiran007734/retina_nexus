"""Helpers for returning actionable errors without leaking local internals."""

from __future__ import annotations

from typing import Any


_SENSITIVE_MARKERS = (
    "\\",
    "/",
    ":\\",
    "c:",
    "secret_key",
    "database_url",
    "password",
    "api_key",
    "token",
    "authorization",
)


def safe_error_message(error: Any, fallback: str, limit: int = 240) -> str:
    """Return an exception message only when it cannot expose local details."""
    message = str(error or "").strip()
    lowered = message.lower()
    if not message or len(message) > limit or any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return fallback
    return message

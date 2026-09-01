"""Small structured logging boundary for local and container deployments."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone


request_id_context: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_SAFE_EXTRA_FIELDS = {"event", "request_id", "method", "path", "status", "status_code", "duration_ms", "stage", "screening_id", "error_code", "trust_category", "configuration_version", "missing_capabilities"}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "request_id": getattr(record, "request_id", request_id_context.get()),
        }
        for field in _SAFE_EXTRA_FIELDS - {"event", "request_id"}:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

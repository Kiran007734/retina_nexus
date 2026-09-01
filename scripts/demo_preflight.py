"""Run a non-mutating pre-demo check for local Retina-Nexus services."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.runtime import verify_models  # noqa: E402


def _settings() -> Settings:
    root_env = ROOT / ".env"
    backend_env = ROOT / "backend" / ".env"
    env_file = backend_env if backend_env.is_file() else root_env if root_env.is_file() else None
    return Settings(_env_file=str(env_file)) if env_file else Settings(_env_file=None)


def _get(url: str, accept: str = "application/json") -> tuple[int, dict | None, str | None]:
    try:
        with urlopen(Request(url, headers={"Accept": accept}), timeout=10) as response:
            body = response.read()
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
            return response.status, parsed, None
    except URLError as exc:
        return 0, None, type(exc).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:5173")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model_check = verify_models(_settings(), load_models=True)
    health_status, health_body, health_error = _get(args.backend_url.rstrip("/") + "/api/v1/health")
    ready_status, ready_body, ready_error = _get(args.backend_url.rstrip("/") + "/api/v1/health/ready")
    root_status, _, root_error = _get(args.backend_url.rstrip("/") + "/")
    frontend_status, _, frontend_error = _get(args.frontend_url.rstrip("/"), accept="text/html")
    checks = {
        "python": sys.version.split()[0],
        "fastapi_importable": importlib.util.find_spec("fastapi") is not None,
        "pillow_importable": importlib.util.find_spec("PIL") is not None,
        "backend_health": {"status_code": health_status, "reachable": health_status == 200, "error": health_error},
        "backend_readiness": {"status_code": ready_status, "ready": ready_status == 200, "body": ready_body, "error": ready_error},
        "api_root": {"status_code": root_status, "reachable": root_status == 200, "error": root_error},
        "frontend": {"status_code": frontend_status, "reachable": frontend_status == 200, "error": frontend_error},
        "report_generator": {"status": "AVAILABLE" if importlib.util.find_spec("app.api.routes.reports") else "UNAVAILABLE"},
        "models": model_check,
        "note": "Non-mutating preflight. Optional model failures are surfaced and do not become fabricated evidence.",
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if checks["backend_readiness"]["ready"] and checks["frontend"]["reachable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

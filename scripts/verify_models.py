"""Verify model manifests, checksums, and loadability for a local deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.runtime import verify_models  # noqa: E402


def _settings() -> Settings:
    root_env = ROOT / ".env"
    backend_env = ROOT / "backend" / ".env"
    # Local backend paths are relative to backend/.env. Prefer that file when
    # both files exist; the root .env remains the Docker/production contract.
    env_file = backend_env if backend_env.is_file() else root_env if root_env.is_file() else None
    return Settings(_env_file=str(env_file)) if env_file else Settings(_env_file=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-load", action="store_true", help="Verify presence and manifests without loading model runtimes")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args()
    result = verify_models(_settings(), load_models=not args.no_load, load_optional_models=not args.no_load)
    output = result if args.json else {
        "status": result["status"],
        "required_models_available": result["required_models_available"],
        "models": {name: {key: value for key, value in check.items() if key != "error_code" or value} for name, check in result["models"].items()},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if result["required_models_available"] or (args.no_load and result["models"].get("classifier", {}).get("artifact_present") and result["models"].get("classifier", {}).get("manifest_present") and result["models"].get("classifier", {}).get("checksum_valid") is not False) else 2


if __name__ == "__main__":
    raise SystemExit(main())

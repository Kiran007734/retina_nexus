"""Report whether authorized DRIVE/IDRiD files support evidence modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.evidence.dataset_support import evidence_dataset_support  # noqa: E402


def main() -> int:
    report = {"dataset_support": evidence_dataset_support(), "note": "Capability report only; no annotations were created or inferred."}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

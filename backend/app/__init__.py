"""RETINA-NEXUS backend application package.

The backend is intentionally runnable from ``backend/`` (the documented local
startup directory) as well as from the repository root and the container
image.  Add the nearest project root containing the shared ``ml`` package so
runtime imports resolve consistently in all three layouts.
"""

import sys
from pathlib import Path

for _candidate in Path(__file__).resolve().parents:
    # ``backend/app/ml`` is the backend-side package and intentionally does
    # not contain the shared ``ml.training`` package.  Require the shared
    # training namespace so it is not selected accidentally.
    if (_candidate / "ml" / "training").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

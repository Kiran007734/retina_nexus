"""Build an OOD reference summary from authorized quality feature vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_vectors(path: Path) -> list[dict[str, float]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        records = json.loads(raw)
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list or JSONL records")
    return [{str(key): float(value) for key, value in record.items() if isinstance(value, (int, float))} for record in records if isinstance(record, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RetinaGuard OOD reference from authorized feature vectors")
    parser.add_argument("--input", required=True, type=Path, help="JSON list or JSONL quality feature vectors")
    parser.add_argument("--output", required=True, type=Path, help="Output reference JSON path")
    args = parser.parse_args()
    vectors = load_vectors(args.input)
    if len(vectors) < 2:
        raise SystemExit("At least two authorized reference vectors are required; no reference was created.")
    keys = sorted({key for vector in vectors for key in vector})
    features = {}
    for key in keys:
        values = np.asarray([vector[key] for vector in vectors if key in vector], dtype=np.float64)
        if len(values) < 2:
            continue
        features[key] = {"count": int(len(values)), "mean": round(float(values.mean()), 8), "std": round(max(float(values.std()), 1e-6), 8)}
    if not features:
        raise SystemExit("No feature has at least two numeric observations; no reference was created.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"format": "retinaguard-feature-reference-v1", "features": features}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "vector_count": len(vectors), "feature_count": len(features)}, indent=2))


if __name__ == "__main__":
    main()

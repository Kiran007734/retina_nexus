"""Bounded full-pipeline benchmark on authoritative Messidor-2 originals.

This benchmark composes the existing local screening services without writing
database records or changing any model. It is intentionally bounded and writes
one result atomically after each image so an interrupted run is inspectable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from scripts.benchmark_pipeline import _run_once, _services, _settings  # noqa: E402

MANIFEST = ROOT / "ml" / "evaluation" / "messidor2" / "authoritative_external_manifest.json"
OUTPUT_ROOT = ROOT / "ml" / "evaluation" / "master_final" / "benchmarks"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(values)
    index = int((len(ordered) - 1) * 0.95)
    return {"count": len(values), "mean_ms": statistics.mean(values), "median_ms": statistics.median(values), "p95_ms": ordered[index], "min_ms": min(values), "max_ms": max(values)}


async def main_async(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(manifest.get("records", []))
    if manifest.get("evaluation_image_source") != "ORIGINAL_ONLY":
        raise RuntimeError("Benchmark requires the ORIGINAL_ONLY authoritative manifest")
    records = records[: max(0, args.count)]
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else OUTPUT_ROOT / f"authoritative_{args.count}"
    cache = output / "per_image"
    started = time.perf_counter()
    settings = _settings()
    services = _services(settings)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        elapsed = time.perf_counter() - started
        if elapsed >= args.max_wall_seconds:
            break
        image_path = ROOT / record["image_path"]
        try:
            content = image_path.read_bytes()
            began = time.perf_counter()
            result = await _run_once(content, image_path.name, services)
            result.update({"image_id": record["image_id"], "image_path": record["image_path"], "image_sha256": record["sha256"], "source_manifest": manifest_path.relative_to(ROOT).as_posix(), "elapsed_wall_ms": (time.perf_counter() - began) * 1000.0, "completed_at_utc": now()})
        except Exception as exc:
            result = {"status": "FAILED", "image_id": record["image_id"], "image_path": record["image_path"], "image_sha256": record["sha256"], "source_manifest": manifest_path.relative_to(ROOT).as_posix(), "errors": {"benchmark": {"type": type(exc).__name__, "message": "Benchmark failed; no result was substituted."}}, "completed_at_utc": now()}
        rows.append(result)
        atomic_json(cache / f"{index:04d}_{record['image_id']}.json", result)
        completed = sum(row.get("status") in {"COMPLETED", "QUALITY_BLOCKED"} for row in rows)
        failed = sum(row.get("status") == "FAILED" for row in rows)
        print(json.dumps({"completed_records": len(rows), "requested": len(records), "terminal": completed, "failed": failed, "elapsed_seconds": round(time.perf_counter() - started, 2)}, sort_keys=True), flush=True)
    terminal = [row for row in rows if row.get("status") in {"COMPLETED", "QUALITY_BLOCKED"}]
    stage_values: dict[str, list[float]] = {}
    for row in terminal:
        for stage, value in (row.get("stage_timings_ms") or {}).items():
            if isinstance(value, (int, float)):
                stage_values.setdefault(stage, []).append(float(value))
    summary = {
        "schema_version": "authoritative-messidor-full-pipeline-benchmark-v1",
        "status": "COMPLETE" if len(rows) == len(records) and all(row.get("status") in {"COMPLETED", "QUALITY_BLOCKED"} for row in rows) else "PARTIAL_OR_TIMEOUT",
        "generated_at_utc": now(),
        "source_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "requested_count": len(records),
        "processed_count": len(rows),
        "terminal_count": len(terminal),
        "failed_count": sum(row.get("status") == "FAILED" for row in rows),
        "max_wall_seconds": args.max_wall_seconds,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "stage_latency": {stage: stats(values) for stage, values in stage_values.items()},
        "model_or_threshold_changes": False,
        "clinical_validation_claim": False,
        "official_idrid_test_images_opened": 0,
        "rows": rows,
        "note": "Bounded engineering benchmark only. QUALITY_BLOCKED means the quality gate stopped clinical AI for that image; it is not a model prediction.",
    }
    atomic_json(output / "runtime_benchmark.json", summary)
    report = ["# Authoritative Messidor-2 full-pipeline benchmark", "", f"Status: **{summary['status']}**", "", f"- Requested: `{summary['requested_count']}`", f"- Processed: `{summary['processed_count']}`", f"- Terminal: `{summary['terminal_count']}`", f"- Failed: `{summary['failed_count']}`", f"- Elapsed seconds: `{summary['elapsed_wall_seconds']:.3f}`", f"- Max wall seconds: `{summary['max_wall_seconds']}`", "", "This is an engineering benchmark of the existing pipeline. It does not establish clinical performance and does not change models or thresholds.", "", "## Stage latency", "", "```json", json.dumps(summary["stage_latency"], indent=2, sort_keys=True), "```", ""]
    (output / "runtime_benchmark.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "runtime_benchmark.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(output.relative_to(ROOT)), "processed": summary["processed_count"], "terminal": summary["terminal_count"], "stage_latency": summary["stage_latency"]}, indent=2, default=str))
    return 0 if summary["status"] == "COMPLETE" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.count < 1 or args.max_wall_seconds <= 0:
        raise SystemExit("--count must be positive and --max-wall-seconds must be positive")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Resumable CPU runner for the frozen Messidor-2 full pipeline.

This runner changes execution infrastructure only.  It composes the existing
production services in isolated worker processes, caches one terminal result
per image, and never changes model weights, thresholds, or production routes.
It is intentionally safe to interrupt and rerun.

Examples:
    python scripts/run_messidor2_parallel.py --workers 2 --torch-threads 1
    python scripts/run_messidor2_parallel.py --workers 2 --limit 4
    python scripts/run_messidor2_parallel.py --workers 2 --retry-completed
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

OUTPUT_ROOT = ROOT / "ml" / "evaluation" / "messidor2" / "final_external"
SOURCE_RESULTS = OUTPUT_ROOT / "per_image_results.jsonl"
TERMINAL_STATUSES = {"COMPLETED", "QUALITY_BLOCKED"}
RUNNER_VERSION = "messidor2-parallel-full-pipeline-v1"

_WORKER_SERVICES: Any = None
_WORKER_SETTINGS: Any = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cache_key(image_id: str, image_sha256: str, model_fingerprint: str) -> str:
    material = f"{RUNNER_VERSION}|{image_id}|{image_sha256}|{model_fingerprint}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def cache_path(cache_dir: Path, job: dict[str, Any], model_fingerprint: str) -> Path:
    return cache_dir / f"{cache_key(str(job['image_id']), str(job['image_sha256']), model_fingerprint)}.json"


def load_jobs(source: Path = SOURCE_RESULTS) -> list[dict[str, Any]]:
    if not source.is_file():
        raise FileNotFoundError(f"The completed classifier population artifact is missing: {source}")
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", "")).strip()
            if not image_id or image_id.casefold() in seen:
                continue
            seen.add(image_id.casefold())
            jobs.append({
                "image_id": image_id,
                "image_path": str(row["image_path"]),
                "image_sha256": str(row["image_sha256"]),
            })
    if not jobs:
        raise RuntimeError("No valid Messidor-2 jobs were found")
    return jobs


def read_completed(cache_file: Path, job: dict[str, Any], model_fingerprint: str) -> dict[str, Any] | None:
    if not cache_file.is_file():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("job_key") != cache_key(job["image_id"], job["image_sha256"], model_fingerprint):
        return None
    if payload.get("status") not in TERMINAL_STATUSES:
        return None
    return payload


def deduplicate_jobs(jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for job in jobs:
        unique.setdefault(str(job["image_id"]).casefold(), job)
    return list(unique.values())


def progress_snapshot(completed: int, total: int, failures: int, started: float) -> dict[str, Any]:
    elapsed = max(0.0, time.perf_counter() - started)
    rate = completed / elapsed if completed and elapsed else 0.0
    remaining = max(0, total - completed)
    return {
        "completed": completed,
        "total": total,
        "failures": failures,
        "elapsed_seconds": round(elapsed, 3),
        "estimated_remaining_seconds": round(remaining / rate, 3) if rate else None,
        "throughput_images_per_second": round(rate, 6) if rate else None,
    }


def _worker_init(torch_threads: int) -> None:
    global _WORKER_SERVICES, _WORKER_SETTINGS
    from scripts.benchmark_pipeline import _services, _settings

    try:
        import torch

        torch.set_num_threads(max(1, int(torch_threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        # The actual worker call reports a structured failure if ML imports
        # are unavailable; initializer failure should not hide that record.
        pass
    _WORKER_SETTINGS = _settings()
    _WORKER_SERVICES = _services(_WORKER_SETTINGS)


def _run_worker(job: dict[str, Any], model_fingerprint: str) -> dict[str, Any]:
    from scripts.benchmark_pipeline import _run_once

    try:
        image_path = ROOT / "ml" / "datasets" / "raw" / "messidor" / job["image_path"]
        content = image_path.read_bytes()
        result = asyncio.run(_run_once(content, image_path.name, _WORKER_SERVICES))
        result.update({
            "image_id": job["image_id"],
            "image_path": job["image_path"],
            "image_sha256": job["image_sha256"],
            "job_key": cache_key(job["image_id"], job["image_sha256"], model_fingerprint),
            "worker_pid": os.getpid(),
            "runner_version": RUNNER_VERSION,
            "completed_at_utc": utc_now(),
        })
        return result
    except Exception as exc:
        return {
            "status": "FAILED",
            "image_id": job["image_id"],
            "image_path": job["image_path"],
            "image_sha256": job["image_sha256"],
            "job_key": cache_key(job["image_id"], job["image_sha256"], model_fingerprint),
            "worker_pid": os.getpid(),
            "runner_version": RUNNER_VERSION,
            "completed_at_utc": utc_now(),
            "errors": {"worker": {"type": type(exc).__name__, "message": "Worker failed; no prediction was substituted."}},
        }


def aggregate(results: list[dict[str, Any]], manifest: dict[str, Any], output: Path = OUTPUT_ROOT / "full_pipeline_results.json") -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: str(item.get("image_id", "")).casefold())
    statuses: dict[str, int] = {}
    for row in ordered:
        status = str(row.get("status", "FAILED"))
        statuses[status] = statuses.get(status, 0) + 1
    limited_run = bool(manifest.get("limited_run", False))
    payload = {
        "schema_version": "messidor2-full-pipeline-results-v1",
        "generated_at_utc": utc_now(),
        "status": "BENCHMARK_PARTIAL" if limited_run else ("COMPLETE" if len(ordered) == manifest["total_jobs"] and not manifest["failed_jobs"] else "PARTIAL"),
        "run_manifest": "parallel_run_manifest.json",
        "evaluation_scope": "BENCHMARK_SUBSET" if limited_run else "FULL_MESSIDOR2",
        "source_total_jobs": manifest.get("source_total_jobs", manifest["total_jobs"]),
        "limited_run": limited_run,
        "total_jobs": manifest["total_jobs"],
        "result_count": len(ordered),
        "status_counts": statuses,
        "results": ordered,
        "official_test_images_opened": 0,
        "production_promoted": False,
        "clinical_validation_claim": False,
    }
    write_atomic_json(output, payload)
    return payload


def runtime_report(manifest: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row["total_elapsed_ms"]) for row in results if isinstance(row.get("total_elapsed_ms"), (int, float))]
    terminal = [row for row in results if row.get("status") in TERMINAL_STATUSES]
    old_serial_seconds = 70.774 * manifest["total_jobs"]
    measured_mean_seconds = (sum(durations) / len(durations) / 1000.0) if durations else None
    estimated_parallel_seconds = (measured_mean_seconds * manifest["total_jobs"] / max(1, manifest["workers"])) if measured_mean_seconds else None
    measured_elapsed = manifest.get("elapsed_seconds")
    limited_run = bool(manifest.get("limited_run", False))
    return {
        "schema_version": "messidor2-runtime-benchmark-v1",
        "status": "MEASURED_BENCHMARK_SUBSET" if limited_run else ("MEASURED_PARTIAL_RUN" if len(terminal) < manifest["total_jobs"] else "MEASURED_COMPLETE_RUN"),
        "evaluation_scope": "BENCHMARK_SUBSET" if limited_run else "FULL_MESSIDOR2",
        "source_total_jobs": manifest.get("source_total_jobs", manifest["total_jobs"]),
        "limited_run": limited_run,
        "old_serial_estimate_seconds": old_serial_seconds,
        "old_serial_estimate_hours": old_serial_seconds / 3600.0,
        "workers": manifest["workers"],
        "torch_threads_per_worker": manifest["torch_threads_per_worker"],
        "measured_elapsed_seconds": measured_elapsed,
        "completed_sample_count": len(durations),
        "mean_completed_pipeline_seconds": measured_mean_seconds,
        "estimated_parallel_total_seconds": estimated_parallel_seconds,
        "estimated_speedup": old_serial_seconds / estimated_parallel_seconds if estimated_parallel_seconds else None,
        "note": "Parallel total and speedup are engineering estimates until all jobs complete; no clinical or throughput claim is made.",
    }


def write_report(payload: dict[str, Any], runtime: dict[str, Any], path: Path = OUTPUT_ROOT / "full_pipeline_report.md") -> None:
    lines = [
        "# Messidor-2 resumable parallel full-pipeline run",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This runner composes the frozen production services only. It does not tune or replace models, thresholds, preprocessing, or production behavior.",
        "",
        f"- Evaluation scope: **{payload['evaluation_scope']}** (`{payload['result_count']}` of `{payload['source_total_jobs']}` source jobs).",
        f"- Jobs: `{payload['result_count']}/{payload['total_jobs']}` terminal results.",
        f"- Status counts: `{json.dumps(payload['status_counts'], sort_keys=True)}`.",
        f"- Workers: `{runtime['workers']}`; torch threads per worker: `{runtime['torch_threads_per_worker']}`.",
        f"- Prior serial estimate: `{runtime['old_serial_estimate_hours']:.2f}` hours.",
        f"- Measured elapsed time: `{runtime['measured_elapsed_seconds']}` seconds.",
        f"- Estimated parallel total: `{runtime['estimated_parallel_total_seconds']}` seconds.",
        f"- Estimated speedup: `{runtime['estimated_speedup']}`.",
        "",
        "Failed jobs are isolated under `parallel_failures/`; successful terminal results are cached atomically under `parallel_cache/` and are skipped on resume.",
        "",
        "No official test images were opened, no external labels were used for optimization, and no clinical validation claim is made.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-completed", action="store_true", help="Explicitly recompute completed cache entries")
    parser.add_argument("--model-fingerprint", default="production-default-frozen-artifacts")
    args = parser.parse_args()
    if args.workers < 1 or args.torch_threads < 1:
        raise SystemExit("--workers and --torch-threads must be positive")
    jobs = deduplicate_jobs(load_jobs())
    source_total_jobs = len(jobs)
    limited_run = args.limit is not None
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]
    cache_dir = OUTPUT_ROOT / "parallel_cache"
    failure_dir = OUTPUT_ROOT / "parallel_failures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        existing = None if args.retry_completed else read_completed(cache_path(cache_dir, job, args.model_fingerprint), job, args.model_fingerprint)
        if existing is not None:
            results.append(existing)
        else:
            pending.append(job)
    manifest: dict[str, Any] = {
        "schema_version": "messidor2-parallel-run-v1",
        "runner_version": RUNNER_VERSION,
        "generated_at_utc": utc_now(),
        "status": "RUNNING",
        "evaluation_scope": "BENCHMARK_SUBSET" if limited_run else "FULL_MESSIDOR2",
        "source_total_jobs": source_total_jobs,
        "limited_run": limited_run,
        "total_jobs": len(jobs),
        "cached_jobs": len(results),
        "pending_jobs": len(pending),
        "completed_jobs": len(results),
        "failed_jobs": 0,
        "workers": args.workers,
        "torch_threads_per_worker": args.torch_threads,
        "model_fingerprint": args.model_fingerprint,
        "official_test_images_opened": 0,
        "production_promoted": False,
        "clinical_validation_claim": False,
    }
    write_atomic_json(OUTPUT_ROOT / "parallel_run_manifest.json", manifest)
    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=_worker_init, initargs=(args.torch_threads,)) as executor:
            futures = {executor.submit(_run_worker, job, args.model_fingerprint): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "FAILED", "image_id": job["image_id"], "image_path": job["image_path"], "image_sha256": job["image_sha256"], "job_key": cache_key(job["image_id"], job["image_sha256"], args.model_fingerprint), "errors": {"future": {"type": type(exc).__name__, "message": "Future failed; no prediction was substituted."}}}
                results.append(result)
                if result.get("status") in TERMINAL_STATUSES:
                    write_atomic_json(cache_path(cache_dir, job, args.model_fingerprint), result)
                else:
                    write_atomic_json(failure_dir / f"{job['image_id']}.json", result)
                completed = sum(item.get("status") in TERMINAL_STATUSES for item in results)
                failures = sum(item.get("status") == "FAILED" for item in results)
                snapshot = progress_snapshot(completed, len(jobs), failures, started)
                manifest.update({"completed_jobs": completed, "failed_jobs": failures, "last_progress": snapshot, "elapsed_seconds": snapshot["elapsed_seconds"]})
                write_atomic_json(OUTPUT_ROOT / "parallel_run_manifest.json", manifest)
                print(json.dumps(snapshot, sort_keys=True), flush=True)
    completed = sum(item.get("status") in TERMINAL_STATUSES for item in results)
    failures = sum(item.get("status") == "FAILED" for item in results)
    run_status = "BENCHMARK_PARTIAL" if limited_run else ("COMPLETE" if completed == len(jobs) and failures == 0 else "PARTIAL")
    manifest.update({"status": run_status, "completed_jobs": completed, "failed_jobs": failures, "elapsed_seconds": round(time.perf_counter() - started, 3)})
    write_atomic_json(OUTPUT_ROOT / "parallel_run_manifest.json", manifest)
    aggregate_payload = aggregate(results, manifest)
    runtime = runtime_report(manifest, results)
    write_atomic_json(OUTPUT_ROOT / "runtime_benchmark.json", runtime)
    write_report(aggregate_payload, runtime)
    print(json.dumps({"status": manifest["status"], "completed": completed, "total": len(jobs), "failed": failures, "elapsed_seconds": manifest["elapsed_seconds"], "runtime_benchmark": runtime}, indent=2, default=str), flush=True)
    return 0 if manifest["status"] in {"COMPLETE", "BENCHMARK_PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
